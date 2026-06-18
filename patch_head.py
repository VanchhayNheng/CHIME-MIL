import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchMILHead(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=256, num_classes=2, dropout=0.3):
        super().__init__()
        self.attention_v = nn.Linear(input_dim, hidden_dim)
        self.attention_u = nn.Linear(input_dim, hidden_dim)
        self.attention_w = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, num_classes),
        )

    def forward(self, patch_feats):
        gated = torch.tanh(self.attention_v(patch_feats)) * torch.sigmoid(self.attention_u(patch_feats))
        attn_logits = self.attention_w(gated).squeeze(-1)  # [B, N], pre-softmax
        attn = F.softmax(attn_logits, dim=1)               # [B, N]
        slide_emb = (patch_feats * attn.unsqueeze(-1)).sum(dim=1)
        cls_logits = self.classifier(slide_emb)
        return slide_emb, cls_logits, attn, attn_logits
