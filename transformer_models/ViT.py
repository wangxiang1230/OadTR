import torch
import torch.nn as nn
import torch.nn.functional as F
from .decoder import Decoder, DecoderLayer
from .attn import FullAttention, ProbAttention, AttentionLayer
from .Transformer import TransformerModel
from ipdb import set_trace
from .PositionalEncoding import (
    FixedPositionalEncoding,
    LearnedPositionalEncoding,
)

__all__ = ['ViT_B16', 'ViT_B32', 'ViT_L16', 'ViT_L32', 'ViT_H14']


class VisionTransformer_v3(nn.Module):
    def __init__(
        self,
        args,
        img_dim,
        patch_dim,
        out_dim,
        embedding_dim,
        num_heads,
        num_layers,
        hidden_dim,
        dropout_rate=0.0,
        attn_dropout_rate=0.0,
        use_representation=True,
        conv_patch_representation=False,
        positional_encoding_type="learned", with_camera=True, with_motion=True, num_channels=3072,
    ):
        super(VisionTransformer_v3, self).__init__()

        assert embedding_dim % num_heads == 0
        assert img_dim % patch_dim == 0
        self.with_camera = with_camera
        self.with_motion = with_motion
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.patch_dim = patch_dim
        # num_channels = img_dim
        self.num_channels = num_channels
        self.dropout_rate = dropout_rate
        self.attn_dropout_rate = attn_dropout_rate
        self.conv_patch_representation = conv_patch_representation

        # self.num_patches = int((img_dim // patch_dim) ** 2)
        self.num_patches = int(img_dim // patch_dim)
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
        print('position encoding :', positional_encoding_type)

        self.pe_dropout = nn.Dropout(p=self.dropout_rate)

        self.encoder = TransformerModel(
            embedding_dim,
            num_layers,
            num_heads,
            hidden_dim,
            self.dropout_rate,
            self.attn_dropout_rate,
        )
        self.pre_head_ln = nn.LayerNorm(embedding_dim)

        d_model = args.decoder_embedding_dim
        use_representation = False  # False
        if use_representation:
            self.mlp_head = nn.Sequential(
                nn.Linear(embedding_dim + d_model, hidden_dim//2),
                # nn.Tanh(),
                nn.ReLU(),
                nn.Linear(hidden_dim//2, out_dim),
            )
        else:
            self.mlp_head = nn.Linear(embedding_dim + d_model, out_dim)

        if self.conv_patch_representation:
            # self.conv_x = nn.Conv2d(
            #     self.num_channels,
            #     self.embedding_dim,
            #     kernel_size=(self.patch_dim, self.patch_dim),
            #     stride=(self.patch_dim, self.patch_dim),
            #     padding=self._get_padding(
            #         'VALID', (self.patch_dim, self.patch_dim),
            #     ),
            # )
            self.conv_x = nn.Conv1d(
                self.num_channels,
                self.embedding_dim,
                kernel_size=self.patch_dim,
                stride=self.patch_dim,
                padding=self._get_padding(
                    'VALID',  (self.patch_dim),
                ),
            )
        else:
            self.conv_x = None

        self.to_cls_token = nn.Identity()

        # Decoder
        factor = 1  # 5
        dropout = args.decoder_attn_dropout_rate
        # d_model = args.decoder_embedding_dim
        n_heads = args.decoder_num_heads
        d_layers = args.decoder_layers
        d_ff = args.decoder_embedding_dim_out  # args.decoder_embedding_dim_out or 4*args.decoder_embedding_dim None
        activation = 'gelu'  # 'gelu'
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(FullAttention(True, factor, attention_dropout=dropout),  # True
                                   d_model, n_heads),  # ProbAttention  FullAttention
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout),  # False
                                   d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.decoder_cls_token = nn.Parameter(torch.zeros(1, args.query_num, d_model))
        if positional_encoding_type == "learned":
            self.decoder_position_encoding = LearnedPositionalEncoding(
                args.query_num, self.embedding_dim, args.query_num
            )
        elif positional_encoding_type == "fixed":
            self.decoder_position_encoding = FixedPositionalEncoding(
                self.embedding_dim,
            )
        print('position decoding :', positional_encoding_type)
        self.classifier = nn.Linear(d_model, out_dim)
        self.after_dropout = nn.Dropout(p=self.dropout_rate)
        # self.merge_fc = nn.Linear(d_model, 1)
        # self.merge_sigmoid = nn.Sigmoid()

    def forward(self, sequence_input_rgb, sequence_input_flow):
        if self.with_camera and self.with_motion:
            x = torch.cat((sequence_input_rgb, sequence_input_flow), 2)
        elif self.with_camera:
            x = sequence_input_rgb
        elif self.with_motion:
            x = sequence_input_flow

        x = self.linear_encoding(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        # x = torch.cat((cls_tokens, x), dim=1)
        x = torch.cat((x, cls_tokens), dim=1)
        x = self.position_encoding(x)
        x = self.pe_dropout(x)   # not delete

        # apply transformer
        x = self.encoder(x)
        x = self.pre_head_ln(x)  # [128, 33, 1024]
        # x = self.after_dropout(x)  # add
        # decoder
        decoder_cls_token = self.decoder_cls_token.expand(x.shape[0], -1, -1)
        # decoder_cls_token = self.after_dropout(decoder_cls_token)  # add
        # decoder_cls_token = self.decoder_position_encoding(decoder_cls_token)  # [128, 8, 1024]
        dec = self.decoder(decoder_cls_token, x)   # [128, 8, 1024]
        dec = self.after_dropout(dec)  # add
        # merge_atte = self.merge_sigmoid(self.merge_fc(dec))  # [128, 8, 1]
        # dec_for_token = (merge_atte*dec).sum(dim=1)  # [128, 1024]
        # dec_for_token = (merge_atte*dec).sum(dim=1)/(merge_atte.sum(dim=-2) + 0.0001)
        dec_for_token = dec.mean(dim=1)
        # dec_for_token = dec.max(dim=1)[0]
        dec_cls_out = self.classifier(dec)
        # set_trace()
        # x = self.to_cls_token(x[:, 0])
        x = torch.cat((self.to_cls_token(x[:, -1]), dec_for_token), dim=1)
        x = self.mlp_head(x)
        # x = F.log_softmax(x, dim=-1)

        return x, dec_cls_out

    def _get_padding(self, padding_type, kernel_size):
        assert padding_type in ['SAME', 'VALID']
        if padding_type == 'SAME':
            _list = [(k - 1) // 2 for k in kernel_size]
            return tuple(_list)
        return tuple(0 for _ in kernel_size)


def ViT_B16(dataset='imagenet'):
    if dataset == 'imagenet':
        img_dim = 224
        out_dim = 1000
        patch_dim = 16
    elif 'cifar' in dataset:
        img_dim = 32
        out_dim = 10
        patch_dim = 4

    return VisionTransformer(
        img_dim=img_dim,
        patch_dim=patch_dim,
        out_dim=out_dim,
        num_channels=3,
        embedding_dim=768,
        num_heads=12,
        num_layers=12,
        hidden_dim=3072,
        dropout_rate=0.1,
        attn_dropout_rate=0.0,
        use_representation=False,
        conv_patch_representation=False,
        positional_encoding_type="learned",
    )


def ViT_B32(dataset='imagenet'):
    if dataset == 'imagenet':
        img_dim = 224
        out_dim = 1000
        patch_dim = 32
    elif 'cifar' in dataset:
        img_dim = 32
        out_dim = 10
        patch_dim = 4

    return VisionTransformer(
        img_dim=img_dim,
        patch_dim=patch_dim,
        out_dim=out_dim,
        num_channels=3,
        embedding_dim=768,
        num_heads=12,
        num_layers=12,
        hidden_dim=3072,
        dropout_rate=0.1,
        attn_dropout_rate=0.0,
        use_representation=False,
        conv_patch_representation=False,
        positional_encoding_type="learned",
    )


def ViT_L16(dataset='imagenet'):
    if dataset == 'imagenet':
        img_dim = 224
        out_dim = 1000
        patch_dim = 16
    elif 'cifar' in dataset:
        img_dim = 32
        out_dim = 10
        patch_dim = 4

    return VisionTransformer(
        img_dim=img_dim,
        patch_dim=patch_dim,
        out_dim=out_dim,
        num_channels=3,
        embedding_dim=1024,
        num_heads=16,
        num_layers=24,
        hidden_dim=4096,
        dropout_rate=0.1,
        attn_dropout_rate=0.0,
        use_representation=False,
        conv_patch_representation=False,
        positional_encoding_type="learned",
    )


def ViT_L32(dataset='imagenet'):
    if dataset == 'imagenet':
        img_dim = 224
        out_dim = 1000
        patch_dim = 32
    elif 'cifar' in dataset:
        img_dim = 32
        out_dim = 10
        patch_dim = 4

    return VisionTransformer(
        img_dim=img_dim,
        patch_dim=patch_dim,
        out_dim=out_dim,
        num_channels=3,
        embedding_dim=1024,
        num_heads=16,
        num_layers=24,
        hidden_dim=4096,
        dropout_rate=0.1,
        attn_dropout_rate=0.0,
        use_representation=False,
        conv_patch_representation=False,
        positional_encoding_type="learned",
    )


def ViT_H14(dataset='imagenet'):
    if dataset == 'imagenet':
        img_dim = 224
        out_dim = 1000
        patch_dim = 14
    elif 'cifar' in dataset:
        img_dim = 32
        out_dim = 10
        patch_dim = 4

    return VisionTransformer(
        img_dim=img_dim,
        patch_dim=patch_dim,
        out_dim=out_dim,
        num_channels=3,
        embedding_dim=1280,
        num_heads=16,
        num_layers=32,
        hidden_dim=5120,
        dropout_rate=0.1,
        attn_dropout_rate=0.0,
        use_representation=False,
        conv_patch_representation=False,
        positional_encoding_type="learned",
    )
