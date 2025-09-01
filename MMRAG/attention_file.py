import torch
import torch.nn as nn



class CrossModalAttentionN(nn.Module):
    def __init__(self, emb_dim = 512):
        super().__init__()
        self.emb_dim = emb_dim

        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),
        )


    def forward(self, txt_emb, img_emb):
        combined = torch.cat([txt_emb, img_emb], dim=-1)

        fused_emb = self.mlp(combined)

        return fused_emb

