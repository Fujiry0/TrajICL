
import os

import numpy as np
import pandas as pd

from utils.trajnetplusplustools import Reader_jrdb_2dbox, Reader_jta_all_visual_cues


def load_data_jrdb_2dbox(split, r, data_dir="data"):
    joint_and_mask = []
    # change dataset path
    name = "jrdb_2dbox"
    train_scenes, _, _ = prepare_data(
        f"{data_dir}/jrdb_2dbox/", subset=split, sample=1.0, goals=False, dataset_name=name
    )  # train_scenes: list of trajectories
    filename_list = []
    frames_list = []
    pedestrians_list = []
    for scene_i, (filename, scene_id, paths) in enumerate(train_scenes):  # trajctories
        scene_train, frames, pedestrians = Reader_jrdb_2dbox.paths_to_xy(
            paths
        )  # (21, N, 8) (21, 28, 8)
        filename_list.append(filename)
        frames_list.append(frames)
        pedestrians_list.append(pedestrians)
        scene_train = drop_ped_with_missing_frame(
            scene_train
        )  # 観測フレームにnanが入っていたら削除
        if r is not None:
            scene_train, _ = drop_distant_far(scene_train)  # (21, n, 8)

        scene_train_real = scene_train.reshape(
            scene_train.shape[0], scene_train.shape[1], -1, 4
        )  # (21, n, 2, 4)
        scene_train_real_ped = np.transpose(
            scene_train_real, (1, 0, 2, 3)
        )  # (n, 21, 2, 4)

        scene_train_mask = np.ones(scene_train_real_ped.shape[:-1])  # (n, 21, 2)
        joint_and_mask.append(
            (
                np.asarray(scene_train_real_ped)[:, :, 0:1, :3],  # (n, 21, 1, 3)
                np.asarray(scene_train_mask)[:, :, 0:1],  # (n, 21, 2)
            )
        )

    return joint_and_mask, filename_list, frames_list, pedestrians_list


def load_data_jta_all_visual_cues(split, r=6, data_dir="data"):
    joint_and_mask = []
    # change dataset path
    name = "jta_all_visual_cues"

    train_scenes, _, _ = prepare_data(
        f"{data_dir}/jta_all_visual_cues/",
        subset=split,
        sample=1.0,
        goals=False,
        dataset_name=name,
    )
    filename_list = []
    frames_list = []
    pedestrians_list = []
    for scene_i, (filename, scene_id, paths) in enumerate(train_scenes):
        scene_train, frames, pedestrians = Reader_jta_all_visual_cues.paths_to_xy(paths)
        filename_list.append(filename)
        frames_list.append(frames)
        pedestrians_list.append(pedestrians)
        scene_train = drop_ped_with_missing_frame(scene_train)
        if r is not None:
            scene_train, _ = drop_distant_far(scene_train, r=r)
        scene_train_real = scene_train.reshape(
            scene_train.shape[0], scene_train.shape[1], -1, 4
        )
        scene_train_real_ped = np.transpose(scene_train_real, (1, 0, 2, 3))

        scene_train_mask = np.ones(scene_train_real_ped.shape[:-1])
        joint_and_mask.append(
            (
                np.asarray(scene_train_real_ped)[:, :, 0:1, :3],
                np.asarray(scene_train_mask)[:, :, 0:1],
            )
        )

    return joint_and_mask, filename_list, frames_list, pedestrians_list


def prepare_data(path, subset="/train/", sample=1.0, goals=True, dataset_name=""):
    all_scenes = []

    # List file names
    files = [
        f.split(".")[-2] for f in os.listdir(path + subset) if f.endswith(".ndjson")
    ]
    # Iterate over file names
    if dataset_name == "jta_all_visual_cues":
        for file in files:
            reader = Reader_jta_all_visual_cues(
                path + subset + "/" + file + ".ndjson", scene_type="paths"
            )
            scene = [(file, s_id, s) for s_id, s in reader.scenes(sample=sample)]
            all_scenes += scene
        return all_scenes, None, True
    elif dataset_name == "jrdb_2dbox":
        for file in files:
            reader = Reader_jrdb_2dbox(
                path + subset + "/" + file + ".ndjson", scene_type="paths"
            )
            scene = [(file, s_id, s) for s_id, s in reader.scenes(sample=sample)]
            all_scenes += scene
        return all_scenes, None, True
    else:
        print("not implement this dataset, error from utils/data.py")
        exit()


def drop_ped_with_missing_frame(xy):
    xy_n_t = np.transpose(xy, (1, 0, 2))
    mask = np.ones(xy_n_t.shape[0], dtype=bool)
    for n in range(xy_n_t.shape[0] - 1):
        for t in range(9):
            if np.isnan(xy_n_t[n + 1, t, 0]):
                mask[n + 1] = False
                break
    return np.transpose(xy_n_t[mask], (1, 0, 2))


def drop_distant_far(xy, r=6):
    distance_2 = np.sum(np.square(xy[:, :, 0:2] - xy[:, 0:1, 0:2]), axis=2)
    mask = np.nanmin(distance_2, axis=0) < r**2
    return xy[:, mask], mask


# motsynth ###
def load_motsynth(split, r=50, resize=1, stride=21, data_dir="data"):

    with open(f"{data_dir}/motsynth/motsynth_{split}.txt", "r") as file:
        static_scenes = [line.strip() for line in file]

    trajectories, filename_list, frames_list, pedestrians_list = (
        prepare_data_motsynth(static_scenes, stride=stride, data_dir=data_dir)
    )

    joint_and_mask = []

    for scene_i, scene_train in enumerate(trajectories):  # trajctories
        scene_train = scene_train[:, :, 2:]  # (21, 14, 4) -> (21, 14, 2)
        scene_train_tmp = np.zeros(
            [scene_train.shape[0], scene_train.shape[1], 8]
        )  # (21, N, 8)
        scene_train_tmp[:, :, :2] = scene_train
        scene_train = scene_train_tmp
        if r is not None:
            scene_train, _ = drop_distant_far(scene_train, r=r)  # (21, n, 8)
        scene_train_real = scene_train.reshape(
            scene_train.shape[0], scene_train.shape[1], -1, 4
        )  # (21, n, 2, 4)
        scene_train_real_ped = np.transpose(
            scene_train_real, (1, 0, 2, 3)
        )  # (n, 21, 2, 4)
        scene_train_mask = np.ones(scene_train_real_ped.shape[:-1])
        joint_and_mask.append(
            (
                np.asarray(scene_train_real_ped)[:, :, 0:1, :3] * resize,
                np.asarray(scene_train_mask)[:, :, 0:1],
            ),
        )

    return joint_and_mask, filename_list, frames_list, pedestrians_list


def make_motsynth_df(file_path):

    # Define column names for the DataFrame
    column_names = [
        "frame",
        "id",
        "bb_left",
        "bb_top",
        "bb_width",
        "bb_height",
        "conf",
        "x",
        "y",
        "z",
    ]

    # Read the gt.txt file into a pandas DataFrame
    df = pd.read_csv(file_path, header=None, names=column_names, index_col=False)
    df["bb_center_x"] = df["bb_left"] + (df["bb_width"] / 2)
    df["bb_center_y"] = df["bb_top"] + (df["bb_height"] / 2)
    df = df[["frame", "id", "bb_center_x", "bb_center_y"]]

    return df


def prepare_data_motsynth(
    static_scenes,
    step=10,
    seq_len=21,
    stride=21,
    data_dir="data",
):

    trajectories = []
    filename_list = []
    frames_list = []
    pedestrians_list = []

    for static_scene in static_scenes:
        file_path = f"{data_dir}/motsynth/mot_annotations/{static_scene}/gt/gt.txt"
        df = make_motsynth_df(file_path)  # trackId, frame, x, y, sceneId, rec&trackId
        frames = sorted(np.unique(df["frame"]).tolist())
        frames = [i for i in range(frames[0], frames[-1] + 1)]
        frames = frames[::step]  # downsample 30 fps -> 2.5 fps
        frame_data = []
        for frame in frames:
            df_frame = df.loc[df["frame"] == int(frame), :]
            frame_data.append(df_frame)
        frames_len = len(frames)
        n_chunk = (frames_len - seq_len) // stride + 1
        for idx in range(0, n_chunk * stride + 1, stride):
            curr_seq_frames = frames[idx : idx + seq_len]
            if len(curr_seq_frames) != seq_len:
                continue
            curr_seq_data = np.concatenate(frame_data[idx : idx + seq_len], axis=0)
            frames_in_curr_seq = np.unique(curr_seq_data[:, 0])
            if len(frames_in_curr_seq) != seq_len:
                continue
            peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
            trajectory = []
            for _, ped_id in enumerate(peds_in_curr_seq):
                curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]
                if len(curr_ped_seq) != seq_len:
                    continue
                trajectory.append(curr_ped_seq[None])
            if len(trajectory) > 0:
                trajectory = np.concatenate(trajectory, axis=0).transpose(1, 0, 2)
                for n in range(trajectory.shape[1]):
                    trajectory_data = np.concatenate(
                        [
                            trajectory[:, n:].copy(),
                            trajectory[:, :n].copy(),
                        ],
                        axis=1,
                    )
                    trajectories.append(trajectory_data)
                    filename_list.append(static_scene)
                    frames_list.append(curr_seq_frames)
                    pedestrians_list.append(trajectory_data[0, 0, 1])

    return trajectories, filename_list, frames_list, pedestrians_list
