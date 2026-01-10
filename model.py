import warnings

import torch
import torch.nn as nn

from decoder import Decoder
from embedding import DecoderEmbedding
from encoder import Encoder

warnings.simplefilter("ignore")


class Model(nn.Module):
    def __init__(self, cfg):
        super(Model, self).__init__()
        self.dim_hidden = cfg["model"]["dim_hidden"]
        self.activation = cfg["model"]["activation"]
        self.dec_emb = DecoderEmbedding(
            self.dim_hidden,
            dropout=0.1,
            device=cfg["device"],
        )
        self.rel_pos_emb = nn.Linear(2, self.dim_hidden)
        self.encoder = Encoder(
            dim_hidden=self.dim_hidden,
            nhead=cfg["model"]["num_heads"],
            dim_feedfwd=cfg["model"]["dim_feedforward"],
            nlayers_local=cfg["model"]["num_enlayers_local"],
            nlayers_global=cfg["model"]["num_enlayers_global"],
            output_scale=cfg["model"]["output_scale"],
            device=cfg["device"],
            activation=self.activation,
        )

        self.hist_len = cfg["model"]["hist_len"]
        self.fut_len = cfg["model"]["fut_len"]

        self.fc_in_traj = nn.Linear(2, self.dim_hidden)

        self.num_future = cfg["model"]["num_future"]
        self.predict_head_traj = []
        # Multiple prediction heads for trajectory
        for _ in range(self.num_future):
            self.predict_head_traj.append(nn.Linear(self.dim_hidden, 2, bias=False))
        self.predict_head_traj = nn.ModuleList(self.predict_head_traj)

        self.decoder = Decoder(
            hidden_dim=self.dim_hidden,
            num_heads=cfg["model"]["num_heads"],
            dim_feedforward=cfg["model"]["dim_feedforward"],
            num_layers=cfg["model"]["num_denlayers"],
            hist_len=cfg["model"]["hist_len"],
            fut_len=cfg["model"]["fut_len"],
            fc_in_traj=self.fc_in_traj,
            predict_head_traj=self.predict_head_traj,
            num_future=self.num_future,
            activation=self.activation,
        )

        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(
        self,
        hist_traj=None,
        fut_traj=None,
        padding_mask=None,
        example_primary_rel_pos=None,
        training=True,
    ):
        """
        Args:
            hist_traj: Past trajectory, shape [B, C, hist_len, N, 2]
            fut_traj: Past trajectory, shape [B, C, fut_len, N, 2]
            padding_mask: Padding Mask, shape [B, C, N]

        Returns:
            out_fut_pred_traj: Predicted trajectory, shape [B, seq_len, 2]
        """
        B, C, hist_len, N, _ = hist_traj.shape
        B, C, fut_len, N, _ = fut_traj.shape
        seq_len = hist_len + fut_len

        fut_traj[:, -1] = hist_traj[:, -1, -1].unsqueeze(1).repeat(1, fut_len, 1, 1)

        hist_traj = hist_traj.reshape(B * C, hist_len, N, 2)
        fut_traj = fut_traj.reshape(B * C, fut_len, N, 2)
        padding_mask = padding_mask.reshape(B * C, N)

        hist_traj_feat = self.fc_in_traj(
            hist_traj
        )  # torch.Size([B*C, hist_len, N, 2]) ->  torch.Size([B*C, hist_len, N, D])

        fut_traj_feat = self.fc_in_traj(
            fut_traj
        )  # torch.Size([B*C, fut_len, N, 2]) ->  torch.Size([B*C, fut_len, N, D])

        traj_feat = torch.cat(
            [hist_traj_feat, fut_traj_feat], dim=1
        )  # [B*C, seq_len, N, D]

        traj_feat = self.encoder(
            traj_feat, padding_mask
        )  # [N, seq_len, B*C, D]

        traj_feat = traj_feat.permute(2, 1, 0, 3)  # [B*C, seq_len, N, D]

        traj_feat = traj_feat[:, :, 0]  # [B*C, seq_len, D]

        _, seq_len, D = traj_feat.shape
        traj_feat = traj_feat.reshape(B, C, seq_len, D)

        rel_pos = self.rel_pos_emb(
            example_primary_rel_pos
        )  # [B, C, 2] -> [B, C, D]
        traj_feat = traj_feat + rel_pos

        primary_pred_fut_traj = self.decoder(
            traj_feat, training
        )  # [B, C, seq_len, D] -> [B, K_modal, fut_len, 2] or [B, K_modal, total_seq_len, 2]

        output = {
            "primary_pred_fut_traj": primary_pred_fut_traj,  # [B, K_modal, fut_len, 2]
        }

        return output


def create_model(cfg):
    model = Model(cfg).to(cfg["device"])

    return model


if __name__ == "__main__":

    from calflops import calculate_flops
    from omegaconf import OmegaConf

    yaml_file_path = "configs/config.yaml"

    cfg = OmegaConf.load(yaml_file_path)
    cfg = OmegaConf.create(cfg)  # Ensure cfg is mutable
    # Disable structure enforcement
    OmegaConf.set_struct(cfg, False)
    cfg.model.num_denlayers = 3
    cfg.training.phase = "finetune"

    B = 1
    N = 10
    C = 1

    hist_len = cfg["model"]["hist_len"]
    fut_len = cfg["model"]["fut_len"]

    # Create input tensors
    hist_traj = torch.randn(B, C, hist_len, N, 2).cuda()
    fut_traj = torch.randn(B, C, fut_len, N, 2).cuda()
    padding_mask = torch.zeros(B, C, N).cuda()
    example_primary_rel_pos = torch.randn(B, C, 2).cuda()

    # Initialize and run the Decoder
    model = Model(cfg).cuda().eval()  # Adjust for partial teacher forcing
    print(model)

    inputs = {
        "hist_traj": hist_traj,
        "fut_traj": fut_traj,
        "padding_mask": padding_mask,
        "example_primary_rel_pos": example_primary_rel_pos,
    }

    flops, macs, params = calculate_flops(
        model=model,
        kwargs=inputs,
        output_as_string=True,
        output_precision=4,
    )

    # Display results
    print("FLOPs: %s   MACs: %s   Params: %s" % (flops, macs, params))
