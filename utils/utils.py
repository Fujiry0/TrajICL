import logging
import os
import random
import sys

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf
from torch.nn.utils.rnn import pad_sequence


def path_to_repo(*args):  # REPO/arg1/arg2
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), *args)


def path_to_data(*args):  # REPO/data/arg1/arg2
    return path_to_repo("data", *args)


def path_to_experiment(*args):  # REPO/experiments/arg1/arg2
    return path_to_repo("experiments", *args)


def path_to_config(*args):  # REPO/configs/arg1/arg2
    return path_to_repo("configs", *args)


def create_logger(logdir):
    head = "%(asctime)-15s %(message)s"
    if logdir != "":
        log_file = os.path.join(logdir, "log.txt")
        logging.basicConfig(filename=log_file, format=head)
        # output to console as well
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    else:
        logging.basicConfig(format=head)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    return logger


def init_output_dirs(exp_name="default"):
    log_dir = path_to_experiment(exp_name)
    ckpt_dir = os.path.join(log_dir, "checkpoints")
    runs_dir = os.path.join(log_dir, "tensorboard")

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(runs_dir, exist_ok=True)

    return log_dir, ckpt_dir, runs_dir


def load_default_config():
    return load_config(path_to_config("default.yaml"))


def load_config(path, exp_name="default"):
    """
    Load the config file and make any dynamic edits.
    """
    with open(path, "rt") as reader:
        config = yaml.load(reader, Loader=yaml.Loader)

    # if "OUTPUT" not in config:
    #     config["OUTPUT"] = {}
    # (
    #     config["OUTPUT"]["log_dir"],
    #     config["OUTPUT"]["ckpt_dir"],
    #     config["OUTPUT"]["runs_dir"],
    # ) = init_output_dirs(exp_name=exp_name)

    # with open(os.path.join(config["OUTPUT"]["ckpt_dir"], "config.yaml"), "w") as f:
    #     yaml.dump(config, f)

    return config


class AverageMeter(object):
    """
    From https://github.com/mkocabas/VIBE/blob/master/lib/core/trainer.py
    Keeps track of a moving average.
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def shuffle(hist_trajs, in_masks, ids_ratio=0):
    if ids_ratio == 0:
        return hist_trajs
    for i, (hist_traj, in_mask) in enumerate(zip(hist_trajs, in_masks)):
        for j, _ in enumerate(hist_traj[:-1]):  # shulle except the last observed frame
            valid_num_person = int(torch.sum(in_mask[j]).item())
            ids = torch.where(torch.rand((valid_num_person)) > (1 - ids_ratio))[0]
            shuffled_ids = ids[torch.randperm(len(ids))]
            hist_trajs[i][j][ids] = hist_trajs[i][j][shuffled_ids]

    return hist_trajs


# Error Location
def noising(
    in_joints,
    hist_masks,
    range_noise_dev=False,
    noise_dev=0,
    noise_ratio=0,
):
    B, hist_len, N, _ = in_joints.shape
    incomplete_in_joints = in_joints.clone()

    # Denoising
    if range_noise_dev:
        noise_dev = np.random.uniform(low=0.1, high=noise_dev)
    noise_mask = torch.rand(B, hist_len, N).cuda()
    noise_mask = noise_mask < noise_ratio
    noise_mask *= hist_masks.bool()
    # noise = noise_max_distance * torch.rand(in_joints.shape).float()
    noise = torch.from_numpy(
        np.random.normal(loc=0.0, scale=noise_dev, size=in_joints.shape)
    ).type_as(in_joints)
    noise = noise.cuda() * noise_mask[:, :, :, None]
    incomplete_in_joints = incomplete_in_joints + noise

    return incomplete_in_joints, noise_mask


# Miss-Detection
def masking(
    in_joints,
    hist_masks,
    masking_ratio=0,
):
    B, hist_len, N, _ = in_joints.shape
    incomplete_in_joints = in_joints.clone()

    # Masking
    masking_mask = torch.ones((B, hist_len * N))
    shuffle_indices = torch.rand((B, hist_len * N)).argsort().cuda()
    mask_indices = shuffle_indices[:, : int(N * hist_len * (1 - masking_ratio))]
    batch_ind = torch.arange(B)[:, None].cuda()
    masking_mask[batch_ind, mask_indices] = 0
    masking_mask = masking_mask.view(B, hist_len, N).cuda() * hist_masks
    incomplete_in_joints[masking_mask.bool()] = 0

    return incomplete_in_joints, masking_mask


# ID Switching
def idswitch(person_encoding, padding_mask, B, hist_len=9, idswitch_ratio=0.1):
    error_person_encoding = person_encoding.repeat_interleave(B, dim=0)
    if idswitch_ratio == 0:
        return error_person_encoding
    for i, (_, mask) in enumerate(zip(error_person_encoding, padding_mask)):
        valid_num_person = int(torch.sum(~mask.to(torch.bool)).item())
        for j in range(hist_len - 1):
            shuffle_ids = torch.where(
                torch.rand((valid_num_person)) > (1.0 - idswitch_ratio)
            )[0]
            error_person_encoding[i][j][shuffle_ids] = error_person_encoding[i][j][
                shuffle_ids[torch.randperm(len(shuffle_ids))]
            ]
    return error_person_encoding


# shorten hist length
def shorten(in_joints, hist_masks, min_hist_len=2, max_hist_len=8):
    B, hist_len, N, _ = in_joints.shape
    incomplete_in_joints = in_joints.clone()

    new_hist_len = random.choice(range(min_hist_len, max_hist_len + 1))
    shorte_mask = torch.zeros((B, hist_len, N)).cuda()
    incomplete_in_joints[
        :,
        :-new_hist_len,
    ] = 0
    shorte_mask[
        :,
        :-new_hist_len,
    ] = 1
    shorte_mask *= hist_masks

    return incomplete_in_joints, shorte_mask


def add_random_false_positive_per_timestep(
    hist_all_traj, padding_mask, ratio=0.1, noise_scale=0.5
):
    B, hist_len, N, _ = hist_all_traj.shape
    device = (
        hist_all_traj.device
    )  # Ensure everything runs on the same device (e.g., cuda)

    # Create masks for valid and invalid indices
    valid_indices = padding_mask == 0  # Shape: [B, N]
    invalid_indices = padding_mask == 1  # Shape: [B, N]

    new_hist_all_traj = []
    new_padding_mask = []
    for b in range(B):
        num_valid = valid_indices[b].sum().item()
        num_invalid = invalid_indices[b].sum().item()

        curr_hist_all_traj = hist_all_traj[b]
        curr_padding_mask = padding_mask[b]

        if num_valid == 0:
            new_hist_all_traj.append(curr_hist_all_traj)
            new_padding_mask.append(curr_padding_mask)
            continue

        # Calculate the number of false positives to add for this batch
        num_false_positives = int(num_valid * ratio)

        if num_false_positives <= 0:
            new_hist_all_traj.append(curr_hist_all_traj)
            new_padding_mask.append(curr_padding_mask)
            continue  # If no false positives to add, skip this batch

        # Generate random noise and ensure it's on the correct device
        noise = torch.tensor(
            np.random.normal(
                loc=0.0, scale=noise_scale, size=(num_false_positives, hist_len, 2)
            ),
            dtype=torch.float32,
            device=device,
        )

        # If there are enough invalid positions, randomly add false positives
        if num_invalid >= num_false_positives:
            # Randomly select invalid positions to place false positives
            selected_invalid_indices = (
                invalid_indices[b].nonzero(as_tuple=False).squeeze(1)
            )
            selected_invalid_indices = selected_invalid_indices[
                torch.randperm(num_invalid)[:num_false_positives]
            ]

            # Randomly sample valid positions for the same timestep to replace as false positives
            random_valid_indices = torch.multinomial(
                valid_indices[b].float(),
                num_false_positives * hist_len,
                replacement=True,
            ).view(hist_len, num_false_positives)

            # Use torch.gather to select random positions from the same timestep
            for t in range(hist_len):
                selected_valid_positions = hist_all_traj[
                    b, t, random_valid_indices[t], :
                ]
                hist_all_traj[b, t, selected_invalid_indices, :] = (
                    selected_valid_positions + noise[:, t, :]
                )

            # Update the padding_mask to mark these positions as valid (set to 0)
            padding_mask[b, selected_invalid_indices] = 0
            curr_hist_all_traj = hist_all_traj[b]
            curr_padding_mask = padding_mask[b]
        # If there are not enough invalid positions, use available ones and expand the rest
        else:
            if num_invalid > 0:
                # Use all available invalid positions
                selected_invalid_indices = (
                    invalid_indices[b].nonzero(as_tuple=False).squeeze(1)
                )
                random_valid_indices = torch.multinomial(
                    valid_indices[b].float(), num_invalid * hist_len, replacement=True
                ).view(hist_len, num_invalid)

                # Replace invalid positions with random false positive trajectories
                # from different positions per timestep
                for t in range(hist_len):
                    selected_valid_positions = hist_all_traj[
                        b, t, random_valid_indices[t], :
                    ]
                    hist_all_traj[b, t, selected_invalid_indices, :] = (
                        selected_valid_positions + noise[:num_invalid, t, :]
                    )

                padding_mask[b, selected_invalid_indices] = 0

            curr_hist_all_traj = hist_all_traj[b]
            curr_padding_mask = padding_mask[b]

            # Calculate how many more false positives are needed
            remaining_false_positives = num_false_positives - num_invalid

            if remaining_false_positives > 0:
                # Generate random noise for the additional false positives
                additional_noise = torch.tensor(
                    np.random.normal(
                        loc=0.0,
                        scale=noise_scale,
                        size=(remaining_false_positives, hist_len, 2),
                    ),
                    dtype=torch.float32,
                    device=device,
                )

                # Randomly sample valid positions for this specific timestep
                random_valid_indices = torch.multinomial(
                    valid_indices[b].float(),
                    remaining_false_positives * hist_len,
                    replacement=True,
                ).view(hist_len, remaining_false_positives)

                # Create new false positive trajectories from the same timestep
                false_positive_traj = torch.zeros(
                    (hist_len, remaining_false_positives, 2), device=device
                )
                for t in range(hist_len):
                    selected_valid_positions = hist_all_traj[
                        b, t, random_valid_indices[t], :
                    ]
                    false_positive_traj[t, :, :] = (
                        selected_valid_positions + additional_noise[:, t, :]
                    )
                curr_hist_all_traj = torch.cat(
                    [hist_all_traj[b], false_positive_traj], dim=1
                )
                false_positive_mask = torch.zeros(
                    remaining_false_positives, dtype=torch.long, device=device
                )
                curr_padding_mask = torch.cat(
                    [padding_mask[b], false_positive_mask], dim=0
                )

        new_hist_all_traj.append(curr_hist_all_traj)
        new_padding_mask.append(curr_padding_mask)

    new_hist_all_traj = [t.permute(1, 0, 2) for t in new_hist_all_traj]
    new_hist_all_traj = pad_sequence(new_hist_all_traj, padding_value=0).permute(
        1, 2, 0, 3
    )
    new_padding_mask = pad_sequence(new_padding_mask, batch_first=True, padding_value=1)

    return new_hist_all_traj, new_padding_mask


def update_stats(
    stats,
    losses_avg,
    split,
    eval_name="",
):
    if len(eval_name) > 0:
        eval_name = eval_name + "_"

    for key in losses_avg.keys():
        stats[f"{eval_name}{key}/{split}"] = losses_avg[key].avg

    return stats


def load_model_checkpoint(cfg, model, optimizer, scheduler):
    """
    Load model weights and optionally resume training state.
    Args:
        cfg: Configuration object.
        model: Model instance.
        optimizer: Optimizer instance.
    Returns:
        start_epoch: Starting epoch.
        model: Model with loaded weights.
        optimizer: Optimizer with loaded state.
    """
    start_epoch = 0
    model_path = os.path.join(
        cfg.output_dir, cfg.load_model.model_dir, cfg.load_model.model_path
    )
    checkpoint = torch.load(model_path, map_location="cuda")
    # Restore epoch and optimizer state if `resume` is True
    if cfg.load_model.resume:
        model.load_state_dict(checkpoint["model"], strict=False)
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            print("Optimizer loaded.")
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"] + 1  # Start from the next epoch
            print("Epoch loaded.")
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
            print("scheduler loaded.")
        print(f"Resuming training from epoch {start_epoch}.")
    else:
        model.load_state_dict(checkpoint["model"], strict=False)
        print("Loaded model parameters without resuming training.")

    return start_epoch, model, optimizer, scheduler


def get_nb_trainable_parameters(model):
    r"""
    Returns the number of trainable parameters and number of all parameters in the model.
    """
    # note: same as PeftModel.get_nb_trainable_parameters
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        # Due to the design of 4bit linear layers from bitsandbytes
        # one needs to multiply the number of parameters by 2 to get
        # the correct number of parameters
        if param.__class__.__name__ == "Params4bit":
            num_params = num_params * 2

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return trainable_params, all_param


def setup_wandb_logging(cfg):
    """
    Set up wandb for logging.
    Args:
        cfg: Configuration object.
        config_dict: Dictionary of configuration parameters.
    Returns:
        wandb: Wandb object.
        output_dir: Directory for saving outputs.
    """
    config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    if cfg.wandb:
        import wandb

        wandb.init(
            config=config_dict,
            project=cfg.training.project,
        )
        project_name = wandb.run.project
        run_name = wandb.run.name
        output_dir = os.path.join(cfg.output_dir, project_name, run_name)
        os.makedirs(output_dir, exist_ok=True)
    else:
        wandb = None
        output_dir = None

    return wandb, output_dir


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    cfg,
    output_dir,
    filename="last_epoch_checkpoint.pth.tar",
):
    if cfg.wandb:
        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "cfg": cfg,
        }
        if scheduler is not None:
            ckpt["scheduler"] = scheduler.state_dict()

        torch.save(ckpt, os.path.join(output_dir, filename))


def logging_wandb(cfg, model, optimizer, scheduler, epoch, stats, output_dir, wandb):
    save_checkpoint(model, optimizer, scheduler, epoch, cfg, output_dir)
    if cfg.wandb:
        for key, val in stats.items():
            if "ade" in key or "fde" in key:
                val /= cfg.training.resize
            wandb.log(
                {
                    f"{key}": val,
                },
                step=epoch,
            )


def freeze_params(model, freeze_layer="encoder"):
    # Freeze all parameters by default
    for name, param in model.named_parameters():
        if freeze_layer in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    return model
