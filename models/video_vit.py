import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TubeletEmbedding(nn.Module):
    def __init__(self, in_channels=3, embed_dim=512, tubelet_size=(2, 16, 16)):
        super().__init__()
        self.tubelet_size = tubelet_size
        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=tubelet_size,
            stride=tubelet_size,
            bias=True,
        )

    def forward(self, x):
        x = self.proj(x)
        batch_size, channels, t_tokens, h_tokens, w_tokens = x.size()
        x = x.view(batch_size, channels, t_tokens * h_tokens * w_tokens)
        x = x.transpose(1, 2)
        return x


class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (
            1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x))
        )


class MLP(nn.Module):
    def __init__(self, embed_dim, mlp_dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, mlp_dim)
        self.act = GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(mlp_dim, embed_dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError('embed_dim must be divisible by num_heads')

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(float(self.head_dim))

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.out_drop = nn.Dropout(dropout)

    def _reshape_heads(self, x):
        batch_size, token_count, _ = x.size()
        x = x.view(batch_size, token_count, self.num_heads, self.head_dim)
        x = x.transpose(1, 2).contiguous()
        return x

    def forward(self, x):
        batch_size, token_count, _ = x.size()

        q = self._reshape_heads(self.q_proj(x))
        k = self._reshape_heads(self.k_proj(x))
        v = self._reshape_heads(self.v_proj(x))

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_drop(attn_probs)

        context = torch.matmul(attn_probs, v)
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, token_count, self.embed_dim)
        context = self.out_proj(context)
        context = self.out_drop(context)
        return context


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VideoViTBackbone(nn.Module):
    def __init__(
        self,
        input_size=(16, 112, 112),
        tubelet_size=(2, 16, 16),
        in_channels=3,
        embed_dim=512,
        depth=4,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.1,
        with_classifier=False,
        num_classes=101,
    ):
        super().__init__()
        self.with_classifier = with_classifier
        self.num_classes = num_classes
        self.output_dim = embed_dim
        self.input_size = input_size
        self.tubelet_size = tubelet_size

        t, h, w = input_size
        pt, ph, pw = tubelet_size
        if (t % pt) != 0:
            raise ValueError('input temporal size must be divisible by tubelet temporal size')
        if (h % ph) != 0:
            raise ValueError('input height must be divisible by tubelet height')
        if (w % pw) != 0:
            raise ValueError('input width must be divisible by tubelet width')

        self.patch_embed = TubeletEmbedding(
            in_channels=in_channels,
            embed_dim=embed_dim,
            tubelet_size=tubelet_size,
        )

        num_patches = (t // pt) * (h // ph) * (w // pw)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)

        if self.with_classifier:
            self.linear = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
        self.apply(self._init_module)

    def _init_module(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.Conv3d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)

    def forward_features(self, x):
        expected_t, expected_h, expected_w = self.input_size
        if x.dim() != 5:
            raise ValueError('expected 5D input [B, C, T, H, W], got {}'.format(tuple(x.shape)))
        if x.shape[2] != expected_t or x.shape[3] != expected_h or x.shape[4] != expected_w:
            raise ValueError(
                'expected input spatial-temporal size ({}, {}, {}), got ({}, {}, {})'.format(
                    expected_t, expected_h, expected_w, x.shape[2], x.shape[3], x.shape[4]
                )
            )

        x = self.patch_embed(x)
        batch_size = x.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = x[:, 0]
        return x

    def forward(self, x):
        x = self.forward_features(x)
        if self.with_classifier:
            x = self.linear(x)
        return x


if __name__ == '__main__':
    model = VideoViTBackbone(with_classifier=False)
    x = torch.randn(2, 3, 16, 112, 112)
    y = model(x)
    print('output shape:', y.shape)
