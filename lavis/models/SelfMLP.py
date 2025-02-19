import torch
import torch.nn as nn


class ITMProcessorAttention(nn.Module):
    def __init__(self, input_dim, num_heads=4):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, batch_first=True)
        self.fc = nn.Linear(input_dim, 1)  # 变换到标量维度

    def forward(self, x):
        attn_output, _ = self.self_attn(x, x, x)  # Self-Attention
        x = self.fc(attn_output).squeeze(-1)  # 变换到标量维度
        return x.mean(dim=1) * 4 + 1  # 计算最终的 itm_score