import random

import torch
import torch.nn as nn
import torch.nn.functional as F


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m,nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m,nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class MIXSTYLE_AB_MIL(nn.Module):
    """AB_MIL with a MixStyle-style feature-statistics mixing layer for
    cross-center domain generalization (Zhou et al., "Domain Generalization
    with MixStyle", ICLR 2021 -- adapted here for single-bag MIL training).

    Standard MixStyle mixes per-sample feature mean/std between two samples
    drawn from the same mini-batch. Because MIL bags are trained one at a
    time (batch_size=1, variable patch counts), there is no second sample in
    the batch to mix with. Instead this keeps a running (EMA) per-center mean
    and std of the post-projection patch features -- one accumulator per
    center, updated only from that center's own bags -- and, for a given
    training bag, mixes its instance statistics with a *different*,
    randomly-chosen center's running statistics. This exposes the shared
    feature extractor to plausible "what would this bag look like under
    another center's style" perturbations during training, without ever
    needing an adversarial objective, and is disabled at eval time (identity
    pass-through), matching standard MixStyle usage.
    """

    def __init__(self,L = 512,D = 128,num_classes = 2,dropout=0,act= nn.ReLU() ,in_dim = 512,
                 num_domains = 3, mix_prob = 0.5, mix_alpha = 0.1, momentum = 0.9, eps = 1e-6, rrt = None):
        super(MIXSTYLE_AB_MIL, self).__init__()
        self.rrt = rrt
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.num_domains = num_domains
        self.mix_prob = mix_prob
        self.mix_alpha = mix_alpha
        self.momentum = momentum
        self.eps = eps
        self.L = L
        self.D = D
        self.K = 1
        self.feature = [nn.Linear(in_dim, self.L)]
        self.feature += [act]
        if dropout:
            self.feature += [nn.Dropout(dropout)]
        if self.rrt is not None:
            self.feature += [self.rrt]
        self.feature = nn.Sequential(*self.feature)

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, self.num_classes),
        )

        # Running per-center feature statistics. Buffers (not parameters):
        # updated in-place via EMA, saved/loaded with the checkpoint, never
        # touched by the optimizer. `center_seen` tracks whether a center's
        # accumulator has received at least one real update yet, so we don't
        # mix against a still-zero/untouched buffer early in training.
        self.register_buffer('center_mu', torch.zeros(num_domains, L))
        self.register_buffer('center_sigma', torch.ones(num_domains, L))
        self.register_buffer('center_seen', torch.zeros(num_domains, dtype=torch.bool))

        self.apply(initialize_weights)

    def _update_and_maybe_mix(self, feature, domain_id):
        # feature: [N, L] post-projection patch features for this one bag
        mu = feature.mean(dim=0)
        sigma = feature.std(dim=0) + self.eps

        if self.training and domain_id is not None and 0 <= domain_id < self.num_domains:
            with torch.no_grad():
                if not bool(self.center_seen[domain_id]):
                    self.center_mu[domain_id] = mu.detach()
                    self.center_sigma[domain_id] = sigma.detach()
                    self.center_seen[domain_id] = True
                else:
                    m = self.momentum
                    self.center_mu[domain_id] = m * self.center_mu[domain_id] + (1 - m) * mu.detach()
                    self.center_sigma[domain_id] = m * self.center_sigma[domain_id] + (1 - m) * sigma.detach()

            other_domains = [d for d in range(self.num_domains) if d != domain_id and bool(self.center_seen[d])]
            if other_domains and random.random() < self.mix_prob:
                target = random.choice(other_domains)
                lam = float(torch.distributions.Beta(self.mix_alpha, self.mix_alpha).sample())
                mixed_mu = lam * mu + (1 - lam) * self.center_mu[target]
                mixed_sigma = lam * sigma + (1 - lam) * self.center_sigma[target]
                normalized = (feature - mu) / sigma
                return normalized * mixed_sigma + mixed_mu

        return feature

    def forward(self, x, domain_id = None, return_WSI_attn = False, return_WSI_feature = False):
        forward_return = {}
        feature = self.feature(x)
        feature = feature.squeeze(0)
        feature = self._update_and_maybe_mix(feature, domain_id)
        A = self.attention(feature)
        A_ori = A.clone()
        A = torch.transpose(A, -1, -2)
        A = F.softmax(A, dim=-1)
        M = torch.mm(A, feature)
        logits = self.classifier(M)
        forward_return['logits'] = logits
        if return_WSI_feature:
            forward_return['WSI_feature'] = M
        if return_WSI_attn:
            forward_return['WSI_attn'] = A_ori
        return forward_return
