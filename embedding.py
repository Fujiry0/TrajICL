import torch
import torch.nn as nn


class EncoderEmbedding(nn.Module):
    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        device="cuda:0",
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.device = device
        self.time_encoding = nn.Embedding(1000, d_model // 2, max_norm=True).to(device)
        self.person_encoding = nn.Embedding(1000, d_model // 2, max_norm=True).to(
            device
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: torch.Size([16, 21, 18, 128])
        _, seq_len, N, _ = x.shape

        half = x.size(3) // 2  # 64
        # self.learned_encoding(torch.arange(seq_len)): torch.Size([F, D//2])
        # x[:, :, :, 0 : half * 2 : 2]: torch.Size([B, F, N, D//2])
        x[:, :, :, 0 : half * 2 : 2] = x[
            :, :, :, 0 : half * 2 : 2
        ] + self.time_encoding(torch.arange(seq_len).to(self.device)).unsqueeze(
            1
        ).unsqueeze(
            0
        )  # torch.Size([B, F, N, D//2]) + torch.Size([1,  F, N, 1])

        # self.person_encoding(torch.arange(N)): torch.Size([N, D//2])
        person_encoding = self.person_encoding(
            torch.arange(N)
            .unsqueeze(0)
            .repeat_interleave(seq_len, dim=0)
            .to(self.device)
        ).unsqueeze(
            0
        )  # torch.Size([1, 21, 18, 64])

        x[:, :, :, 1 : half * 2 : 2] = (
            x[:, :, :, 1 : half * 2 : 2] + person_encoding
        )  # torch.Size([16, 21, 18, 64]) +　torch.Size([1, 18, 64])

        return self.dropout(x)


class TimeEncoderEmbedding(nn.Module):
    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        device="cuda:0",
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.device = device
        self.time_encoding = nn.Embedding(1000, d_model // 2, max_norm=True).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: torch.Size([16, 21, 18, 128])
        _, seq_len, _, D = x.shape

        half = D // 2  # 64
        # self.learned_encoding(torch.arange(seq_len)): torch.Size([F, D//2])
        # x[:, :, :, 0 : half * 2 : 2]: torch.Size([B, F, N, D//2])
        x[:, :, :, 0 : half * 2 : 2] = x[
            :, :, :, 0 : half * 2 : 2
        ] + self.time_encoding(torch.arange(seq_len).to(self.device)).unsqueeze(
            1
        ).unsqueeze(
            0
        )  # torch.Size([B, F, N, D//2]) + torch.Size([1,  F, N, 1])

        return self.dropout(x)


class PersonEncoderEmbedding(nn.Module):
    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        device="cuda:0",
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.device = device
        self.person_encoding = nn.Embedding(1000, d_model // 2, max_norm=True).to(
            device
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: torch.Size([16, 21, 18, 128])
        _, seq_len, N, _ = x.shape

        half = x.size(3) // 2  # 64
        # shuffle the person encoding except the first one
        sampled_indices = torch.randint(1, 1000, (N - 1,)).to(self.device)
        shuffled_indices = torch.cat(
            [torch.tensor([0], device=self.device), sampled_indices]
        )  # Ensure index 0 is 0

        person_encoding = self.person_encoding(
            shuffled_indices.unsqueeze(0).repeat_interleave(seq_len, dim=0)
        ).unsqueeze(
            0
        )  # torch.Size([1, 21, 18, 64])

        x[:, :, :, 1 : half * 2 : 2] = (
            x[:, :, :, 1 : half * 2 : 2] + person_encoding
        )  # torch.Size([16, 21, 18, 64]) +　torch.Size([1, 18, 64])

        return self.dropout(x)


class DecoderEmbedding(nn.Module):
    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        device="cuda:0",
        depos_type="emb",
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.device = device
        self.depos_type = depos_type

        self.emb = nn.Embedding(1000, d_model // 2, max_norm=True).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, d_model].

        Returns:
            torch.Tensor: Tensor with positional encoding added, same shape as input.
        """

        if len(x.shape) == 4:
            _, C, seq_len, D = (
                x.shape
            )  # Remove unused variables B and C # 16, 8, 21, 128
            half = D // 2  # Correctly calculate half based on the last dimension
            x[:, :, :, 0 : half * 2 : 2] = x[:, :, :, 0 : half * 2 : 2] + self.emb(
                torch.arange(C - 1, -1, -1).to(
                    self.device
                )  # Ensure embedding matches dimension
            ).unsqueeze(0).unsqueeze(2)

        if len(x.shape) == 3:
            _, C, seq_len, N, D = (
                x.shape
            )  # Remove unused variables B and C # 16, 8, 21, 128
            half = D // 2  # Correctly calculate half based on the last dimension
            x[:, :, :, :, 0 : half * 2 : 2] = x[:, :, :, :, 0 : half * 2 : 2] + self.emb(
                torch.arange(C - 1, -1, -1, -1).to(
                    self.device
                )  # Ensure embedding matches dimension
            ).unsqueeze(0).unsqueeze(2)

        return self.dropout(x)


# test code for DecoderEmbedding
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder_embedding = DecoderEmbedding(
        d_model=128, device=device,
    )
    x = torch.randn(16, 8, 21, 128).to(device)
    output = decoder_embedding(x)
