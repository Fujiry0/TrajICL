import torch
import torch.nn as nn


class TransformerEncoder(nn.TransformerEncoder):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__(
            encoder_layer=encoder_layer, num_layers=num_layers, norm=norm
        )

    def forward(
        self,
        src,
        src_mask=None,
        src_key_padding_mask=None,
    ):
        output = src

        for mod in self.layers:
            output = mod(
                    output, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask
                )

        if self.norm is not None:
            output = self.norm(output)

        return output


class Decoder(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_heads,
        dim_feedforward,
        num_layers,
        hist_len,
        fut_len,
        fc_in_traj,
        predict_head_traj,
        dropout=0.1,
        activation="relu",
        num_future=20,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.hist_len = hist_len
        self.fut_len = fut_len
        self.seq_len = self.hist_len + self.fut_len
        self.num_future = num_future
        decoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
            )

        self.decoder = TransformerEncoder(decoder_layer, num_layers=num_layers)

        self.fc_in_traj = fc_in_traj
        self.predict_head_traj = predict_head_traj

    def forward(self, seq_feat, training=True):
        B, C, seq_len, D = seq_feat.shape
        seq_feat = seq_feat.reshape(B, C * seq_len, D)  # [B, total_seq_len, hidden_dim]
        B, total_seq_len, D = seq_feat.shape
        seq_feat = seq_feat.permute(1, 0, 2)  # [total_seq_len, B, hidden_dim]
        outputs = self.decoder(
            src=seq_feat,
        )  # [total_seq_len, B, hidden_dim]

        outputs = outputs.permute(1, 0, 2)  # [B, total_seq_len, hidden_dim]

        outputs = outputs[:, -self.fut_len :]

        out_traj = []
        for k in range(self.num_future):
            temp_out = self.predict_head_traj[k](outputs)
            out_traj.append(temp_out)
        out_traj = torch.stack(out_traj)  # [K_modal, B, total_seq_len, hidden_dim]
        outputs = out_traj.permute(1, 0, 2, 3)  # [B, K_modal, seq_len, 2]
        outputs = outputs[:, :, -self.fut_len :]

        return outputs  # [B, K_modal, fut_len, 2] or [B, K_modal, total_seq_len, 2]


if __name__ == "__main__":
    B, hidden_dim = 4, 256
    C = 3
    dim_feedforward = 512
    num_heads = 8
    num_layers = 4
    hist_len = 9
    fut_len = 12
    seq_len = hist_len + fut_len
    num_future = 20

    seq_feat = torch.randn(B, C, seq_len, hidden_dim)

    predict_head_traj = []
    # Multiple prediction heads for trajectory
    for _ in range(num_future):
        predict_head_traj.append(nn.Linear(hidden_dim, 2, bias=False))
    predict_head_traj = nn.ModuleList(predict_head_traj)

    decoder = Decoder(
        hidden_dim,
        num_heads,
        dim_feedforward,
        num_layers,
        hist_len,
        fut_len,
        fc_in_traj=nn.Linear(2, hidden_dim),
        num_future=num_future,
        predict_head_traj=predict_head_traj
    )

    output, output_pi = decoder(seq_feat)

    print("Output shape:", output.shape)  # Expected: [B, num_future, fut_len, 2]

    if output_pi is not None:
        print("Output pi shape:", output_pi.shape)
