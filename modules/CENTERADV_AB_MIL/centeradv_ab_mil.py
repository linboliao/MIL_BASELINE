import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m,nn.Linear):
            # ref from clam
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m,nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class GradReverse(Function):
    """Gradient reversal layer (Ganin & Lempitsky, 2015). Identity on the
    forward pass; negates (and scales by lambd) the incoming gradient on the
    backward pass, so anything upstream of this layer is pushed to *hurt* the
    loss computed downstream of it instead of helping it."""

    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None


def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)


class CENTERADV_AB_MIL(nn.Module):
    """AB_MIL with a center-adversarial (DANN-style) branch for cross-center
    domain generalization.

    The shared bag-level representation M is fed both to the normal
    diagnosis classifier and, through a gradient-reversal layer, to a small
    domain classifier that predicts which center (hospital) the slide came
    from. The domain classifier itself trains normally to get *better* at
    telling centers apart; because its gradient is negated before it reaches
    the shared feature extractor, the feature extractor is instead pushed to
    become *worse* for that task, i.e. to make M center-invariant, while
    still needing to stay predictive of the diagnosis label. This directly
    targets center-level covariate shift (e.g. a held-out hospital such as
    301) rather than post-hoc calibrating the output probabilities.
    """

    def __init__(self,L = 512,D = 128,num_classes = 2,dropout=0,act= nn.ReLU() ,in_dim = 512,
                 num_domains = 3, domain_hidden = 64, rrt = None):
        super(CENTERADV_AB_MIL, self).__init__()
        self.rrt = rrt
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.num_domains = num_domains
        self.L = L
        self.D = D
        self.K = 1
        self.feature = [nn.Linear(in_dim, self.L)]

        self.feature += [act]

        if dropout:
            self.feature += [nn.Dropout(dropout)]

        if self.rrt != None:
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
        self.domain_classifier = nn.Sequential(
            nn.Linear(self.L*self.K, domain_hidden),
            nn.ReLU(),
            nn.Linear(domain_hidden, num_domains),
        )

        self.apply(initialize_weights)

    def forward(self, x, grl_lambda = 0.0, return_WSI_attn = False, return_WSI_feature = False):
        forward_return = {}
        feature = self.feature(x)
        feature = feature.squeeze(0)
        A = self.attention(feature)
        A_ori = A.clone()
        A = torch.transpose(A, -1, -2)  # KxN
        A = F.softmax(A, dim=-1)  # softmax over N
        M = torch.mm(A, feature)  # 1,KxL
        logits = self.classifier(M)
        domain_logits = self.domain_classifier(grad_reverse(M, grl_lambda))
        forward_return['logits'] = logits
        forward_return['domain_logits'] = domain_logits
        if return_WSI_feature:
            forward_return['WSI_feature'] = M
        if return_WSI_attn:
            forward_return['WSI_attn'] = A_ori
        return forward_return
