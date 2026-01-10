import pickle
from collections import defaultdict
from typing import List, Tuple

import torch


def load_processed_data(
    split,
    name,
):
    save_dir = f"processed_data/{name}"
    # split = "val"
    trajs = torch.load(f"{save_dir}/{split}_trajs.pt")
    masks = torch.load(f"{save_dir}/{split}_masks.pt")
    with open(
        f"{save_dir}/{split}_filename2idxs_dict.pickle",
        mode="br",
    ) as fi:
        filename2idxs_dict = pickle.load(fi)

    with open(
        f"{save_dir}/{split}_idx2filename_dict.pickle",
        mode="br",
    ) as fi:
        idx2filename_dict = pickle.load(fi)

    with open(
        f"{save_dir}/{split}_pool_indices_by_fold.pickle",
        mode="br",
    ) as fi:
        pool_indices_by_fold = pickle.load(fi)

    with open(
        f"{save_dir}/{split}_valid_indices_by_fold.pickle",
        mode="br",
    ) as fi:
        valid_indices_by_fold = pickle.load(fi)

    with open(
        f"{save_dir}/{split}_similar_traj_dicts_hist.pickle",
        mode="br",
    ) as fi:
        similarity_dicts = pickle.load(fi)

    similarity_dicts_seq = None
    with open(
        f"{save_dir}/{split}_similar_traj_dicts_seq.pickle",
            mode="br",
    ) as fi:
        similarity_dicts_seq = pickle.load(fi)

    return (
        trajs,
        masks,
        filename2idxs_dict,
        idx2filename_dict,
        pool_indices_by_fold,
        valid_indices_by_fold,
        similarity_dicts,
        similarity_dicts_seq,
    )


def split_indices_by_appearance(
    filename_list: List[str],
    frames_list: List[List[int]],
    pedestrians_ids_list: List[int],
    train_ratio: float = 0.8,
) -> Tuple[List[int], List[int]]:
    """
    ファイルごとに登場順に基づいて train/test のインデックスを取得する関数

    Args:
        filename_list (List[str]): 各データのファイル名リスト
        frames_list (List[List[int]]): 各データのフレーム番号リスト
        pedestrians_ids_list (List[int]): 各データの歩行者IDリスト
        train_ratio (float): 訓練データの割合（デフォルト 0.8）

    Returns:
        train_indices (List[int]): train に属するデータのインデックス
        test_indices (List[int]): test に属するデータのインデックス
    """

    # ファイルごとに、歩行者 ID の最初の登場フレームを取得
    file_to_pedestrians = defaultdict(dict)

    for i in range(len(filename_list)):
        fname = filename_list[i]
        pid = pedestrians_ids_list[i]
        if pid not in file_to_pedestrians[fname]:
            file_to_pedestrians[fname][pid] = min(
                frames_list[i]
            )  # 最初の登場フレームを記録

    # 各ファイルごとに歩行者 ID を登場順でソートし、train/test に分割
    train_pedestrians_per_file = {}
    test_pedestrians_per_file = {}

    for fname, ped_dict in file_to_pedestrians.items():
        sorted_pedestrians = sorted(
            ped_dict.keys(), key=lambda pid: ped_dict[pid]
        )  # 登場順に並べる
        split_idx = int(len(sorted_pedestrians) * train_ratio)

        train_pedestrians_per_file[fname] = set(sorted_pedestrians[:split_idx])
        test_pedestrians_per_file[fname] = set(sorted_pedestrians[split_idx:])

    # インデックスを振り分ける
    train_indices, test_indices = [], []

    for i in range(len(filename_list)):
        fname = filename_list[i]
        pid = pedestrians_ids_list[i]

        if pid in train_pedestrians_per_file[fname]:
            train_indices.append(i)  # 元のインデックスを記録
        else:
            test_indices.append(i)  # 元のインデックスを記録

    return train_indices, test_indices


def create_trajs_masks(data):

    datalist = []

    num_people = []
    for scene in data:
        trajs, mask = scene
        N, T, _, _ = trajs.shape
        num_people.append(N)
        people = []
        for n in range(len(trajs)):
            people.append((torch.from_numpy(trajs[n]), torch.from_numpy(mask[n])))
        datalist.append(people)

    trajs = []
    masks = []

    for scene in datalist:
        traj = torch.stack([s[0] for s in scene])  # torch.Size([N, 21, 1, 3])
        mask = torch.stack([s[1] for s in scene])  # torch.Size([N, 21, 1])
        trajs.append(traj)
        masks.append(mask)

    return trajs, masks, num_people
