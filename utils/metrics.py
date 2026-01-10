import torch


def mse_primary_ade_loss(primary_pred_fut_traj, primary_gt_fut_traj):
    """
    Computes the Average Displacement Error (ADE) between the predicted and ground truth future trajectories.

    Args:
        primary_pred_fut_traj (torch.Tensor): Predicted future trajectory of shape (B, T, 2).
        primary_gt_fut_traj (torch.Tensor): Ground truth future trajectory of shape (B, T, 2).

    Returns:
        torch.Tensor: Mean ADE loss over the batch.
    """
    norm = torch.norm(
        primary_pred_fut_traj - primary_gt_fut_traj, p=2, dim=-1
    )  # [B, T]
    mean = torch.mean(norm, dim=-1)  # [B]
    mean = torch.mean(mean)
    return mean


def mse_primary_fde_loss(primary_pred_fut_traj, primary_gt_fut_traj):
    """
    Computes the Final Displacement Error (FDE) between the predicted and ground truth final positions.

    Args:
        primary_pred_fut_traj (torch.Tensor): Predicted future trajectory of shape (B, T, 2).
        primary_gt_fut_traj (torch.Tensor): Ground truth future trajectory of shape (B, T, 2).

    Returns:
        torch.Tensor: Mean FDE loss over the batch.
    """
    primary_pred_fut_traj_final = primary_pred_fut_traj[:, -1]
    primary_gt_fut_traj_final = primary_gt_fut_traj[:, -1]
    norm = torch.norm(
        primary_pred_fut_traj_final - primary_gt_fut_traj_final, p=2, dim=-1
    )
    mean = torch.mean(norm)
    return mean


def mse_primary_min_ade_loss(primary_pred_fut_traj, primary_gt_fut_traj):
    """
    Computes the minimum ADE (minADE) across multiple predicted trajectories, selecting the best matching one.

    Args:
        primary_pred_fut_traj (torch.Tensor): Predicted future trajectories of shape (B, K, T, 2).
        primary_gt_fut_traj (torch.Tensor): Ground truth future trajectory of shape (B, T, 2).

    Returns:
        torch.Tensor: Sum of minimum ADE over the batch.
    """
    primary_gt_fut_traj = primary_gt_fut_traj.unsqueeze(1)
    norm = torch.norm(
        primary_pred_fut_traj - primary_gt_fut_traj, p=2, dim=-1
    )  # [B, K, T]
    ade = torch.mean(norm, dim=-1)  # [B, K]
    min_ade, min_ade_idx = torch.min(ade, dim=-1)  # [B]
    min_ade = torch.mean(min_ade)
    return min_ade, min_ade_idx


def mse_primary_min_fde_loss(primary_pred_fut_traj, primary_gt_fut_traj):
    """
    Computes the minimum FDE (minFDE) across multiple predicted trajectories, selecting the best matching one.

    Args:
        primary_pred_fut_traj (torch.Tensor): Predicted future trajectories of shape (B, K, T, 2).
        primary_gt_fut_traj (torch.Tensor): Ground truth future trajectory of shape (B, T, 2).

    Returns:
        torch.Tensor: Sum of minimum FDE over the batch.
    """
    primary_gt_fut_traj = primary_gt_fut_traj.unsqueeze(1)
    norm = torch.norm(
        primary_pred_fut_traj - primary_gt_fut_traj, p=2, dim=-1
    )  # [B, K, T]
    fde = norm[:, :, -1]  # [B, K]
    min_fde, min_fde_idx = torch.min(fde, dim=-1)  # [B]
    min_fde = torch.mean(min_fde)
    return min_fde, min_fde_idx


def feature_transform_reguliarzer(trans):
    d = trans.size()[1]
    Identity = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        Identity = Identity.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - Identity, dim=(1, 2)))
    return loss


if __name__ == "__main__":
    B, fut_len = 3, 12
    primary_pred_fut_traj = torch.randn([B, fut_len, 2])
    primary_gt_fut_traj = torch.randn([B, fut_len, 2])
    print(mse_primary_ade_loss(primary_pred_fut_traj, primary_gt_fut_traj))
    print(mse_primary_fde_loss(primary_pred_fut_traj, primary_gt_fut_traj))

    primary_pred_fut_traj = torch.randn([B, 20, fut_len, 2])
    primary_gt_fut_traj = torch.randn([B, fut_len, 2])
    print(mse_primary_min_ade_loss(primary_pred_fut_traj, primary_gt_fut_traj))
    print(mse_primary_min_fde_loss(primary_pred_fut_traj, primary_gt_fut_traj))
