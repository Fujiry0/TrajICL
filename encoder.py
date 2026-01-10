import warnings

import torch
import torch.nn as nn

from embedding import PersonEncoderEmbedding, TimeEncoderEmbedding

warnings.simplefilter("ignore")


class TransformerEncoder(nn.TransformerEncoder):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__(
            encoder_layer=encoder_layer, num_layers=num_layers, norm=norm
        )

    def forward(self, src, mask=None, src_key_padding_mask=None, get_attn=False):
        output = src

        for i, mod in enumerate(self.layers):
            output = mod(
                output, src_mask=mask, src_key_padding_mask=src_key_padding_mask
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


class Encoder(nn.Module):
    def __init__(
        self,
        dim_hidden=128,
        nhead=4,
        dim_feedfwd=1024,
        nlayers_local=6,
        nlayers_global=3,
        dropout=0.1,
        activation="relu",
        output_scale=1,
        device="cuda:0",
    ):
        super(Encoder, self).__init__()
        self.dim_hidden = dim_hidden
        self.output_scale = output_scale
        self.device = device
        self.nlayers_local = nlayers_local

        self.time_enc_emb = TimeEncoderEmbedding(
            dim_hidden,
            dropout=0.1,
            device=device,
        )
        self.person_enc_emb = PersonEncoderEmbedding(
            dim_hidden,
            dropout=0.1,
            device=device,
        )
        self.nlayers_local = nlayers_local
        encoder_layer_local = nn.TransformerEncoderLayer(
            d_model=self.dim_hidden,
            nhead=nhead,
            dim_feedforward=dim_feedfwd,
            dropout=dropout,
            activation=activation,
        )
        self.local_former = TransformerEncoder(
            encoder_layer_local, num_layers=nlayers_local
        )

        encoder_layer_global = nn.TransformerEncoderLayer(
            d_model=self.dim_hidden,
            nhead=nhead,
            dim_feedforward=dim_feedfwd,
            dropout=dropout,
            activation=activation,
        )
        self.global_former = TransformerEncoder(
            encoder_layer_global, num_layers=nlayers_global
        )

    def forward(self, traj_feat, padding_mask):
        """
        Args:
            traj: Past trajectory, shape [B, seq_len, N, D]
            padding_mask: Padding Mask, shape [B, N]

        Returns:
            out_global: feat, shape [N, seq_len, B, D]
        """

        B, seq_len, N, D = traj_feat.shape

        traj_feat = self.time_enc_emb(traj_feat)  # torch.Size([B, F, N, D])

        traj_feat = torch.transpose(traj_feat, 0, 1).reshape(
            seq_len, -1, self.dim_hidden
        )  # torch.Size([B, F, N, D]) -> # torch.Size([F, B*N, D])

        tgt_padding_mask_local = (
            padding_mask.reshape(-1).unsqueeze(1).repeat_interleave(seq_len, dim=1)
        )  # torch.Size([B, N])->torch.Size([B*N, F])

        out_local = self.local_former(
            traj_feat, mask=None, src_key_padding_mask=tgt_padding_mask_local
        )  # torch.Size([F, B*N, D]) -> # torch.Size([N*T, B, D])
        out_local = out_local * self.output_scale + traj_feat

        out_local = (
            out_local.reshape(seq_len, B, N, self.dim_hidden)
            .permute(2, 0, 1, 3)
            .reshape(-1, B, self.dim_hidden)
        )  # torch.Size([F, B*N, 128]) -> torch.Size([N*T, B, D])

        out_local = out_local.reshape(N, seq_len, B, D).permute(2, 1, 0, 3)
        out_local = self.person_enc_emb(out_local)  # torch.Size([B, F, N, D])
        out_local = out_local.permute(2, 1, 0, 3).reshape(N * seq_len, B, D)

        tgt_padding_mask_global = padding_mask.repeat_interleave(
            seq_len, dim=1
        )  # torch.Size([B, F])-> torch.Size([B, N*F])
        out_global = self.global_former(
            out_local, mask=None, src_key_padding_mask=tgt_padding_mask_global
        )  # torch.Size([N*F, B, D])
        out_global = (
            out_global * self.output_scale + out_local
        )  # torch.Size([N*F, B, D])

        out_global = out_global.reshape(
            N, seq_len, out_global.size(1), self.dim_hidden
        )  # torch.Size([N*F, B, D]) -> torch.Size([N, F, B, D])

        return out_global


if __name__ == "__main__":
    B, seq_len, N, D = 2, 21, 10, 128
    traj_feat = torch.randn(B, seq_len, N, D).cuda()
    padding_mask = torch.randint(0, 2, (B, N), dtype=torch.bool).cuda()
    padding_mask = torch.zeros((B, N), dtype=torch.bool).cuda()

    encoder = Encoder().cuda()
    out_global = encoder(traj_feat, padding_mask)

    print("Output shape:", out_global.shape)
