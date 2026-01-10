import random

from utils.metrics import mse_primary_min_ade_loss, mse_primary_min_fde_loss


def compute_multi_loss(
    cfg,
    hist_trajs,
    fut_trajs,
    example_primary_rel_pos,
    padding_mask,
    model,
    training=True,
):
    """
    Computes the prediction loss and associated metrics (ADE, FDE)
    for the trajectory prediction model.

    This function includes logic for data augmentation (changing the number of examples)
    during training, runs the model, and calculates the minimum prediction error.

    Args:
        cfg (dict): Configuration dictionary, includes 'aug.change_num_example' setting.
        hist_trajs (Tensor): Historical trajectories of all examples in the batch.
        fut_trajs (Tensor): Ground truth future trajectories of all examples in the batch.
        example_primary_rel_pos (Tensor): Relative position information for the primary agent.
        padding_mask (Tensor): Mask indicating valid/invalid examples/time steps.
        model (nn.Module): The trajectory prediction model.
        training (bool, optional): Flag indicating whether the function is run during training.
                                   Defaults to True.

    Returns:
        dict: A dictionary containing the total loss ('loss') used for backpropagation
              and itemized metrics ('loss_ade', 'loss_fde').
    """

    res = {}

    # --- Data Augmentation Logic ---
    if training and cfg.aug.change_num_example:
        # Randomly select a number of examples to exclude (data augmentation)
        invalid_num_example = random.choice(range(cfg.dataset.num_example + 1))
        # Remove the leading 'invalid_num_example' agents from the batch data
        hist_trajs = hist_trajs[:, invalid_num_example:]
        fut_trajs = fut_trajs[:, invalid_num_example:]
        padding_mask = padding_mask[:, invalid_num_example:]
        example_primary_rel_pos = example_primary_rel_pos[:, invalid_num_example:]

    # --- Model Forward Pass ---
    output = model(
        hist_trajs,
        fut_trajs.clone(),  # Use clone() to prevent potential in-place modifications
        padding_mask,
        training=training,
        example_primary_rel_pos=example_primary_rel_pos,
    )

    # Extract the primary agent's predicted future trajectory (usually a multimodal output)
    primary_pred_fut_traj = (
        output["primary_pred_fut_traj"]
    )

    # Extract the ground truth future trajectory for the primary agent
    # Assuming the primary agent is the last agent in the sequence index (e.g., -1)
    # and the first dimension of trajectory coordinates (e.g., 0)
    primary_gt_fut_traj = fut_trajs[
        :, -1, :, 0
    ]

    # --- Loss Calculation ---
    # Calculate Minimum Average Displacement Error (minADE) loss
    # The loss_ade returned here is the loss value, min_ade_idx is the index of the best prediction.
    loss_ade, min_ade_idx = mse_primary_min_ade_loss(primary_pred_fut_traj, primary_gt_fut_traj)

    # Calculate Minimum Final Displacement Error (minFDE) loss
    loss_fde, _ = mse_primary_min_fde_loss(primary_pred_fut_traj, primary_gt_fut_traj)

    # The primary loss used for optimization (usually minADE)
    res["loss"] = loss_ade

    # Record metrics (convert to scalar item() for tracking/logging)
    res["loss_ade"] = loss_ade.item()
    res["loss_fde"] = loss_fde.item()

    return res
