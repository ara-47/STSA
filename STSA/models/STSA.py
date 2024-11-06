import torch
import torch.nn as nn
import torch.nn.functional as F
from STSA.models.TISA import TISAModel
from STSA.models.PositionalEncoding import (
    FixedPositionalEncoding,
    LearnedPositionalEncoding,
)
from STSA.models.SetAttention import SetAttention
import torch
import torch.nn as nn
from torchvision import models

__all__ = ['STSA']


class Spatio_Temporal_Set_Attention(nn.Module):
    def __init__(
        self,
        img_dim,
        patch_dim,
        out_dim,
        head_dim,
        num_channels,
        num_sets,
        embedding_dim,
        num_iters,
        num_layers,
        eps,
        hidden_dim,
        dropout_rate=0.0,
        attn_dropout_rate=0.0,
        use_representation=True,
        conv_patch_representation=False,
        positional_encoding_type="learned",
    ):
        super(Spatio_Temporal_Set_Attention, self).__init__()

        assert img_dim % patch_dim == 0


        self.head_dim = head_dim

        self.embedding_dim = embedding_dim
        self.num_sets = num_sets
        self.num_iters = num_iters
        self.eps = eps
        self.patch_dim = patch_dim
        self.out_dim = out_dim
        self.num_channels = num_channels
        self.dropout_rate = dropout_rate
        self.attn_dropout_rate = attn_dropout_rate
        self.conv_patch_representation = conv_patch_representation

        self.num_patches = int((img_dim // patch_dim) ** 2)
        self.seq_length = self.num_patches + 1
        self.flatten_dim = patch_dim * patch_dim * num_channels
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))

        self.linear_encoding = nn.Linear(self.flatten_dim, embedding_dim)
        if positional_encoding_type == "learned":
            self.position_encoding = LearnedPositionalEncoding(
                self.seq_length, self.embedding_dim, self.seq_length
            )
        elif positional_encoding_type == "fixed":
            self.position_encoding = FixedPositionalEncoding(
                self.embedding_dim,
            )

        self.pe_dropout = nn.Dropout(p=self.dropout_rate)

        self.tisa = TISAModel(
            num_sets,
            head_dim,
            embedding_dim,
            num_layers,
            num_iters,
            eps,
            hidden_dim,
            self.dropout_rate,
        )
        self.pre_head_ln = nn.LayerNorm(embedding_dim)
        if use_representation:
            self.mlp_head = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, out_dim),
            )
        else:
            self.mlp_head = nn.Linear(embedding_dim, out_dim)
        self.mlp_multi = nn.Linear(self.num_patches + 1, self.num_patches)
        self.mlp_single = nn.Linear(self.num_patches, int(self.num_patches/self.num_patches))
        self.mlp_single_forloop = nn.Linear(self.embedding_dim, int(self.embedding_dim))
        self.SISA_norm = nn.LayerNorm(self.embedding_dim)

        self.SISA = SetAttention(self.num_sets, self.head_dim, self.embedding_dim, iters=self.num_iters, eps=self.eps, hidden_dim=hidden_dim)

        if self.conv_patch_representation:
            self.conv_x = nn.Conv2d(
                self.num_channels,
                self.embedding_dim,
                kernel_size=(self.patch_dim, self.patch_dim),
                stride=(self.patch_dim, self.patch_dim),
                padding=self._get_padding(
                    'VALID', (self.patch_dim, self.patch_dim),
                ),
            )
        else:
            self.conv_x = None

        self.to_cls_token = nn.Identity()

    def forward(self, x):
        bs, c, h, w = x.shape
        if self.conv_patch_representation:
            x = self.conv_x(x)
            x = x.permute(0, 2, 3, 1).contiguous()
            x = x.view(x.size(0), -1, self.flatten_dim)
        else:
            x = (
                x.unfold(2, self.patch_dim, self.patch_dim)
                .unfold(3, self.patch_dim, self.patch_dim)
                .contiguous()
            )
            x = x.view(bs, c, -1, self.patch_dim ** 2)
            x = x.permute(0, 2, 3, 1).contiguous()
            x = x.view(x.size(0), -1, self.flatten_dim)
            x = self.linear_encoding(x)


        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.position_encoding(x)
        x = self.pe_dropout(x)

        # TISA Encoder
        x = self.tisa(x).permute(0, 2, 1)
        x = self.mlp_multi(x)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x_outputs = torch.zeros(self.num_patches, bs, self.out_dim).to(device)
        orix1 = self.mlp_single(x).permute(0, 2, 1)
        #SISA Encoder
        for i in range(self.num_patches):
            x1 = self.mlp_single_forloop(torch.unsqueeze(x.permute(2, 0, 1)[i], 1)) + orix1
            x1_norm = self.SISA_norm(x1)
            x1_SISA = self.SISA(x1_norm)+x1
            x1_SISA = self.pre_head_ln(x1_SISA)
            x1_SISA = self.to_cls_token(x1_SISA[:, 0])
            x1_SISA = self.mlp_head(x1_SISA)
            x_outputs[i] = x1_SISA
        return x_outputs

    def _get_padding(self, padding_type, kernel_size):
        assert padding_type in ['SAME', 'VALID']
        if padding_type == 'SAME':
            _list = [(k - 1) // 2 for k in kernel_size]
            return tuple(_list)
        return tuple(0 for _ in kernel_size)


def STSA(dataset='SumMe'):
    if dataset == 'SumMe':
        img_dim = 3
        out_dim = 2
        patch_dim = 1
    elif dataset == 'TVSum':
        img_dim = 3
        out_dim = 2
        patch_dim = 1

    return Spatio_Temporal_Set_Attention(
        img_dim=img_dim,
        patch_dim=patch_dim,
        out_dim=out_dim,
        head_dim=64,
        num_channels=512,
        num_sets=10,
        embedding_dim=768,
        num_iters=5,
        num_layers=4,
        eps=1e-8,
        hidden_dim=3072,
        dropout_rate=0.1,
        attn_dropout_rate=0.0,
        use_representation=True,
        conv_patch_representation=False,
        positional_encoding_type="learned",
    )