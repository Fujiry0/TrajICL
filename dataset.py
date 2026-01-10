import random

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms

from load_data import load_processed_data


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        name="motsynth_loc",
        split="train",
        hist_len=9,
        fut_len=12,
        num_example=0,
        prompting="random",
        pool_ratio=1,
    ):

        self.name = name
        self.split = split
        self.hist_len = hist_len
        self.fut_len = fut_len
        self.seq_len = self.hist_len + self.fut_len
        self.num_example = num_example
        self.prompting = prompting
        self.pool_ratio = pool_ratio
        (
            self.trajs,
            self.masks,
            self.filename2idxs_dict,
            self.idx2filename_dict,
            self.pool_indices_by_fold,
            self.valid_indices_by_fold,
            self.similarity_dicts,
            self.similarity_dicts_seq,
        ) = load_processed_data(
            split,
            name,
        )

        if split == "train":
            self.valid_indices_fold_pairs = []
            for fold, valid_indice in enumerate(self.valid_indices_by_fold):
                for valid_indeice in valid_indice:
                    self.valid_indices_fold_pairs.append((fold, valid_indeice))
        else:
            self.valid_indices_fold_pairs = []
            fold = len(self.pool_indices_by_fold) - 1
            for valid_indeice in self.valid_indices_by_fold[fold]:
                self.valid_indices_fold_pairs.append((fold, valid_indeice))

        if self.split != "train" and self.pool_ratio < 1:
            self.pool_indices_by_fold, self.similarity_dicts = reduce_pool_by_ratio(
                pool_ratio,
                self.filename2idxs_dict,
                self.pool_indices_by_fold,
                self.similarity_dicts,
            )

    def __len__(self):
        return len(self.valid_indices_fold_pairs)

    def __getitem__(self, idx):
        fold, valid_idx = self.valid_indices_fold_pairs[idx]
        traj = self.trajs[valid_idx]  # torch.Size([N, 21, 1, 3])
        mask = self.masks[valid_idx]  # torch.Size([N, 21, 1])

        example_idxs = []
        if self.prompting == "random":
            example_idxs = random_prompting(
                valid_idx, self.num_example, self.similarity_dicts[fold]
            )
        elif self.prompting == "sim":
            example_idxs = sim_prompting(
                    valid_idx, self.num_example, self.similarity_dicts[fold]
                )

        trajs_list = []
        masks_list = []
        for example_idx in example_idxs:
            traj_example = self.trajs[example_idx]
            mask_example = self.masks[example_idx]
            trajs_list.append(traj_example)
            masks_list.append(mask_example)
        trajs_list.append(traj)
        masks_list.append(mask)

        return trajs_list, masks_list


def create_dataset(split, cfg):

    dataset = Dataset(
        name=cfg.dataset.name,
        split=split,
        hist_len=cfg["model"]["hist_len"],
        fut_len=cfg["model"]["fut_len"],
        num_example=cfg["dataset"]["num_example"],
        prompting=cfg["dataset"]["prompting"],
        pool_ratio=1,
    )

    return dataset


def collate_batch(batch):
    """
    Collates a batch of data by padding sequences and creating masks.

    Args:
        batch (list of tuples): List of (joints, masks) tuples.

    Returns:
        tuple: Padded joints, masks, and padding mask tensors.
    """
    trajs_list = []
    masks_list = []
    num_people_list = []
    B = len(batch)
    C = len(batch[0][0])
    for data in batch:
        joints_examples, masks_examples = data[0], data[1]
        for joints_example, masks_example in zip(joints_examples, masks_examples):
            trajs_list.append(joints_example)
            masks_list.append(masks_example)
            num_people_list.append(torch.zeros(joints_example.shape[0]))
    trajs = pad_sequence(trajs_list, batch_first=True)
    masks = pad_sequence(masks_list, batch_first=True)
    padding_mask = pad_sequence(
        num_people_list, batch_first=True, padding_value=1
    ).bool()

    _, N, T, num_joints, D = trajs.shape
    # print(B, C, N, T, num_joints, D)
    trajs = trajs.view(B, C, N, T, num_joints, D)
    masks = masks.view(B, C, N, T, num_joints)
    padding_mask = padding_mask.view(B, C, N)

    return trajs, masks, padding_mask


def batch_process_coords(
    trajs,
    masks,
    padding_mask,
    cfg,
    training=False,
    eval_robust=False,
):

    hist_len, fut_len = (
        cfg["model"]["hist_len"],
        cfg["model"]["fut_len"],
    )

    trajs = trajs[:, :, :, :, 0]
    masks = masks[:, :, :, :, 0]

    B, C, N, T, D = trajs.shape
    trajs = trajs.view(B * C, N, T, D)
    masks = masks.view(B * C, N, T)
    padding_mask = padding_mask.view(B * C, N)

    trajs = trajs.to(cfg["device"])
    masks = masks.to(cfg["device"])
    trajs *= cfg.training.resize

    if training:
        trajs = getRandomRotatePoseTransform(cfg)(trajs)

    trajs = trajs.view(B, C, N, T, D)
    trajs = trajs - trajs[:, -1, 0:1, (hist_len - 1) : hist_len].unsqueeze(
        1
    )  # move to target current frame origin
    example_primary_rel_pos = trajs[:, :, 0, (hist_len - 1) : hist_len]  # B, C, 1, 3
    example_primary_rel_pos = example_primary_rel_pos[:, :, :, :2]  # B, C, 2
    trajs = trajs.view(B * C, N, T, D)
    trajs = (
        trajs - trajs[:, 0:1, (hist_len - 1) : hist_len]
    )  # move to current frame origin

    trajs = trajs.transpose(1, 2)[:, :, :, :2]
    masks = masks.transpose(1, 2)

    hist_trajs = trajs[:, :hist_len].float()
    fut_trajs = trajs[:, hist_len : hist_len + fut_len].float()
    hist_masks = masks[:, :hist_len].float()
    fut_masks = masks[:, hist_len : hist_len + fut_len].float()

    hist_trajs[~hist_masks.bool()] = 0
    fut_trajs[~fut_masks.bool()] = 0
    fut_trajs = torch.nan_to_num(fut_trajs)

    if training or eval_robust:
        if np.random.uniform() < cfg.aug.corrupt.p:
            hist_trajs, _, _ = corrupt(cfg, hist_trajs, hist_masks)
        if np.random.uniform() < cfg.aug.short.p:
            hist_trajs = short(
                cfg, hist_trajs
            )

    hist_trajs = hist_trajs.view(B, C, hist_len, N, 2)
    fut_trajs = fut_trajs.view(B, C, fut_len, N, 2)
    padding_mask = padding_mask.view(B, C, N)
    hist_trajs = hist_trajs.view(B, C, hist_len, N, 2)

    return (
        hist_trajs,
        hist_masks,
        fut_trajs,
        fut_masks,
        example_primary_rel_pos.float(),
        padding_mask.float().to(cfg["device"]),
    )


def corrupt(cfg, hist_trajs, hist_masks):
    B, hist_len, N, _ = hist_trajs.shape

    corrupt_ratio = random.uniform(cfg.aug.corrupt.ratio.min, cfg.aug.corrupt.ratio.max)
    corrupt_shuffle_indices = torch.rand((B, hist_len * N)).argsort().cuda()
    corrupt_indices_len = int(N * hist_len * corrupt_ratio)

    # Noising
    noise_ratio = cfg.aug.corrupt.noise_ratio.ratio
    if cfg.aug.corrupt.noise_ratio.random:
        noise_ratio = random.uniform(0, 1)
    noise_mask = torch.zeros((B, hist_len * N))
    noise_indices_len = int(corrupt_indices_len * noise_ratio)
    noise_indices = corrupt_shuffle_indices[:, :noise_indices_len]
    batch_ind = torch.arange(B)[:, None].cuda()
    noise_mask[batch_ind, noise_indices] = 1.0
    noise_mask = noise_mask.view(B, hist_len, N).cuda() * hist_masks
    noise_dev = cfg.aug.corrupt.noise.noise_dev
    if cfg.aug.corrupt.noise.range_noise_dev:
        noise_dev = np.random.uniform(low=0, high=noise_dev)
    noise = torch.from_numpy(
        np.random.normal(loc=0.0, scale=noise_dev, size=hist_trajs.shape)
    ).type_as(hist_trajs)
    noise = noise.cuda() * noise_mask[:, :, :, None]
    hist_trajs = hist_trajs + noise

    # Masking
    masking_mask = torch.zeros((B, hist_len * N))
    masking_indices = corrupt_shuffle_indices[:, noise_indices_len:corrupt_indices_len]
    masking_mask[batch_ind, masking_indices] = 1
    masking_mask = masking_mask.view(B, hist_len, N).cuda() * hist_masks
    hist_trajs[masking_mask.bool()] = 0

    return hist_trajs, noise_mask, masking_mask

def getRandomRotatePoseTransform(config):
    """
    Performs a random rotation about the origin (0, 0, 0)
    """

    def do_rotate(pose_seq):
        B, F, J, K = pose_seq.shape

        angles = torch.deg2rad(torch.rand(B) * 360)

        rotation_matrix = torch.zeros(B, 3, 3).to(pose_seq.device)

        # rotate around z axis (vertical axis)
        rotation_matrix[:, 0, 0] = torch.cos(angles)
        rotation_matrix[:, 0, 1] = -torch.sin(angles)
        rotation_matrix[:, 1, 0] = torch.sin(angles)
        rotation_matrix[:, 1, 1] = torch.cos(angles)
        rotation_matrix[:, 2, 2] = 1

        rot_pose = torch.bmm(pose_seq.reshape(B, -1, 3).float(), rotation_matrix)
        rot_pose = rot_pose.reshape(pose_seq.shape)
        return rot_pose

    return transforms.Lambda(lambda x: do_rotate(x))


def random_prompting(idx, num_example, similarity_dict):
    example_idxs = random.sample(
        similarity_dict[idx],
        num_example,
    )
    return example_idxs


def sim_prompting(idx, num_example, similarity_dict):
    valid_num_num_example = min(num_example, len(similarity_dict[idx]))
    example_idxs = similarity_dict[idx][:valid_num_num_example]
    return example_idxs[::-1]


def random_drop_out_neighbors(traj, mask, p=0.5):
    """
    Randomly drop out some trajectories with probability p.
    """
    N = traj.shape[0]
    idxs = torch.rand(N) < 1 - p
    idxs[0] = True
    return traj[idxs], mask[idxs]


def reduce_pool_by_ratio(
    pool_ratio, filename_to_indices_dict, pool_indices_by_fold, similarity_dicts
):
    """
    This function reduces the pool of indices in each fold based on a specified ratio.
    It updates the pool indices and the similarity dictionaries accordingly.

    Args:
    pool_ratio (float): The ratio by which to reduce the pool (e.g., 0.5 means reducing the pool to 50%).
    filename_to_indices_dict (dict): A dictionary mapping each scene filename to its associated indices.

    Returns:
    tuple: A tuple containing:
        - Updated pool indices for each fold (list of lists).
        - Updated similarity dictionaries (list of dicts).
    """
    scenes = list(filename_to_indices_dict.keys())

    # Initialize list to store new pool indices for each fold
    new_pool_indices_by_fold = []
    for pool_indices in pool_indices_by_fold:
        new_pool_indices_for_current_fold = []

        # For each scene, select a portion of the indices based on the pool_ratio
        for scene in scenes:
            scene_indices = filename_to_indices_dict[scene]
            # Select the indices from the pool that belong to the current scene
            pool_indices_for_scene = [
                idx for idx in pool_indices if idx in scene_indices
            ]
            # Reduce the pool size based on the pool_ratio
            reduced_pool_indices_for_scene = pool_indices_for_scene[
                : int(len(pool_indices_for_scene) * pool_ratio)
            ]
            new_pool_indices_for_current_fold.extend(reduced_pool_indices_for_scene)

        new_pool_indices_by_fold.append(new_pool_indices_for_current_fold)

    # Initialize list to store the new similarity dictionaries
    new_similarity_dicts = []
    for pool_indices, similarity_dict in zip(
        new_pool_indices_by_fold, similarity_dicts
    ):
        pool_indices_set = set(pool_indices)
        # Filter similarity dictionary to keep only the indices in the reduced pool
        filtered_similarity_dict = {
            key: [idx for idx in value if idx in pool_indices_set]
            for key, value in similarity_dict.items()
        }
        new_similarity_dicts.append(filtered_similarity_dict)

    return new_pool_indices_by_fold, new_similarity_dicts


def short(cfg, corrupt_hist_trajs):
    new_hist_len = random.choice(
        range(
            cfg.aug.short.min_hist_len,
            cfg.aug.short.max_hist_len + 1,
        )
    )
    corrupt_hist_trajs[
        :,
        :cfg.model.hist_len - new_hist_len,
    ] = 0

    return corrupt_hist_trajs
