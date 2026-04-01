import torch
import torch.nn as nn
import torch.nn.functional as F


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class AB_MIL_Clinical(nn.Module):
    def __init__(self, L=512, D=128, num_classes=2, dropout=0, act=nn.ReLU(), in_dim=1024, clinical_dim=8, clinical_weight=0.7, fusion_type='concat'):
        super(AB_MIL_Clinical, self).__init__()
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.L = L
        self.D = D
        self.clinical_weight = clinical_weight
        self.patch_weight = 1.0 - clinical_weight
        self.fusion_type = fusion_type

        self.feature = nn.Sequential(
            nn.Linear(in_dim, L),
            act
        )

        if dropout:
            self.feature.add_module("dropout", nn.Dropout(dropout))

        self.attention = nn.Sequential(
            nn.Linear(L, D),
            nn.Tanh(),
            nn.Linear(D, 1)
        )

        self.clinical_net = nn.Sequential(
            nn.Linear(clinical_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout else nn.Identity(),

            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout else nn.Identity(),

            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout else nn.Identity(),

            nn.Linear(64, 32),
            nn.ReLU(),
        )

        for layer in self.clinical_net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        self.repeat_factor = L // 32

        if self.fusion_type == 'concat':
            L = L + L
        self.classifier = nn.Linear(L, num_classes)

    def forward(self, x, clinical=None, return_attention=False, return_WSI_feature=False):
        forward_return = {}
        batch_size, n_patches, _ = x.shape

        # 病理特征提取
        feature = self.feature(x)

        # 注意力聚合
        A = self.attention(feature)
        A_ori = A.clone()
        A = F.softmax(A, dim=1)

        # 加权聚合病理特征
        M = torch.bmm(A.transpose(1, 2), feature)
        M = M.squeeze(1)
        patch_feature_raw = M.clone()

        # 临床特征处理
        clinical_feat = self.clinical_net(clinical)

        clinical_feat = clinical_feat.unsqueeze(2)  # (batch_size, 32, 1)
        clinical_feat = clinical_feat.repeat(1, 1, self.repeat_factor)  # (batch_size, 32, 16)
        clinical_feat = clinical_feat.view(-1, self.L)  # (batch_size, 512)

        # 病理特征适配
        # patch_feat_adapted = self.patch_adapter(M)

        if self.fusion_type == 'concat':
            fused = torch.cat([M, clinical_feat], dim=1)
        elif self.fusion_type == 'weighted':
            fused = M * self.patch_weight + clinical_feat * self.clinical_weight
        elif self.fusion_type == 'gate':
            combined = torch.cat([M, clinical_feat], dim=1)
            gate = self.gate_net(combined)
            fused = gate * M + (1 - gate) * clinical_feat
            forward_return['gate_values'] = gate

        forward_return['fusion_type'] = self.fusion_type
        forward_return['patch_features'] = M
        forward_return['clinical_features'] = clinical_feat

        # 分类
        logits = self.classifier(fused)
        forward_return['logits'] = logits

        if return_WSI_feature:
            forward_return['WSI_feature'] = patch_feature_raw
        if return_attention:
            forward_return['WSI_attn'] = A_ori

        return forward_return
