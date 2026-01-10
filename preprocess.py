import argparse
import json
import os
import pickle
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

import numpy as np
import torch
from tqdm import tqdm

from load_data import create_trajs_masks
from utils.data import (
    load_data_jrdb_2dbox,
    load_data_jta_all_visual_cues,
    load_motsynth,
)

warnings.simplefilter("ignore")


# ============================================================
#  Common utilities
# ============================================================

def pickle_dump(obj, path):
    with open(path, mode="wb") as f:
        pickle.dump(obj, f)


def compute_stats_traj(trajs):
    """
    Compute statistical measures for a batch of trajectories.
    Returns all statistics as simple float values for JSON serialization.
    """
    hist_len = 9
    primary_trajs = []
    primary_velocities = []

    processed_trajs = []
    for traj in trajs:
        traj = torch.nan_to_num(traj)
        # Normalize trajectories by subtracting the position at hist_len-1
        traj = traj - traj[0:1, hist_len - 1 : hist_len, :]
        processed_trajs.append(traj)

        # Extract the primary trajectory (first agent)
        primary_traj = traj[0, :, :]
        primary_trajs.append(primary_traj)
        primary_velocity = torch.diff(primary_traj, dim=0)
        primary_velocities.append(primary_velocity)

    primary_trajs = torch.stack(primary_trajs).numpy()
    primary_mean = float(np.mean(primary_trajs))
    primary_std = float(np.std(primary_trajs))
    primary_min = float(np.min(primary_trajs))
    primary_max = float(np.max(primary_trajs))

    primary_velocities = torch.stack(primary_velocities).numpy()
    avg_primary_velocity = float(np.mean(primary_velocities))

    processed_trajs = torch.cat(processed_trajs, dim=0).numpy()

    stats = {
        "primary_mean": primary_mean,
        "primary_std": primary_std,
        "primary_min": primary_min,
        "primary_max": primary_max,
        "primary_velocity": avg_primary_velocity,
        "traj_max": float(np.max(processed_trajs)),
        "traj_min": float(np.min(processed_trajs)),
    }
    print("Trajectory Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    return stats


# ============================================================
#  Step1: Load raw data, preprocess, and save
# ============================================================

def load_data(
    split,
    name,
    data_dir="data",
    r=50,
    stride=21,
):
    """Unified version of the original load_data"""

    if name == "motsynth":
        trajs, filename_list, frames_list, pedestrians_list = load_motsynth(
            split, r=r, resize=1, stride=stride, data_dir=data_dir
        )
        print(f"loaded {split} processed data !!!")
    elif name == "jrdb":
        trajs, filename_list, frames_list, pedestrians_list = load_data_jrdb_2dbox(
            split, r=r, data_dir=data_dir
        )
    elif name == "jta":
        trajs, filename_list, frames_list, pedestrians_list = (
            load_data_jta_all_visual_cues(split, r=r, data_dir=data_dir)
        )

    trajs, masks, num_people = create_trajs_masks(trajs)
    print(f"{name} {split}: " + str(len(trajs)))
    print(f"average num people: {np.mean(num_people)}")
    print(f"max num people: {np.max(num_people)}")
    print(f"min num people: {np.min(num_people)}")

    config = {
        "num_trajs": int(len(trajs)),
        "avg_num_people": int(np.mean(num_people)),
        "max_num_people": int(np.max(num_people)),
        "min_num_people": int(np.min(num_people)),
        "r": r,
        "stride": stride,
        "name": name,
        "split": split,
        "stats": compute_stats_traj(trajs),
    }

    filename2idxs_dict = defaultdict(list)
    idx2filename_dict = {}
    for i, filename in enumerate(filename_list):
        filename2idxs_dict[filename].append(i)
        idx2filename_dict[i] = filename

    return (
        trajs,
        masks,
        filename_list,
        frames_list,
        pedestrians_list,
        filename2idxs_dict,
        idx2filename_dict,
        config,
    )


def split_pedestrians_by_ratio(
    file_names, pedestrian_ids, pedestrian_frames, valid_ratio=0.2
):
    """
    For each file, get the first appearance frame of each pedestrian ID,
    then split into train/test according to a given ratio.
    """
    file_pid_to_fframe = defaultdict(dict)

    for i in range(len(file_names)):
        fname = file_names[i]
        pid = pedestrian_ids[i]
        fframe = min(pedestrian_frames[i])
        if pid not in file_pid_to_fframe[fname]:
            file_pid_to_fframe[fname][pid] = fframe
        else:
            file_pid_to_fframe[fname][pid] = min(fframe, file_pid_to_fframe[fname][pid])

    pool_pedestrians_by_file = {}
    valid_pedestrians_by_file = {}

    for fname, ped_dict in file_pid_to_fframe.items():
        sorted_pedestrians = sorted(ped_dict.keys(), key=lambda pid: ped_dict[pid])
        valid_pedestrian_num = int(len(sorted_pedestrians) * valid_ratio)
        pool_pedestrians_by_fold = []
        valid_pedestrians_by_fold = []

        num_splits = int(1 / valid_ratio)

        for i in range(num_splits):
            start_idx = i * valid_pedestrian_num
            end_idx = (
                (i + 1) * valid_pedestrian_num
                if (i + 1) * valid_pedestrian_num <= len(sorted_pedestrians)
                else len(sorted_pedestrians)
            )

            valid_pedestrians = set(sorted_pedestrians[start_idx:end_idx])
            pool_pedestrians = set(sorted_pedestrians) - valid_pedestrians

            pool_pedestrians_by_fold.append(pool_pedestrians)
            valid_pedestrians_by_fold.append(valid_pedestrians)

        pool_pedestrians_by_file[fname] = pool_pedestrians_by_fold
        valid_pedestrians_by_file[fname] = valid_pedestrians_by_fold

    pool_indices_by_fold, valid_indices_by_fold = [], []
    for i in range(num_splits):
        pool_indices = []
        valid_indices = []
        for j in range(len(file_names)):
            fname = file_names[j]
            pid = pedestrian_ids[j]
            if pid in pool_pedestrians_by_file[fname][i]:
                pool_indices.append(j)
            else:
                valid_indices.append(j)
        pool_indices_by_fold.append(pool_indices)
        valid_indices_by_fold.append(valid_indices)

    return pool_indices_by_fold, valid_indices_by_fold


def pool_valid_split(
    filename_list,
    frames_list,
    pedestrians_list,
    filename2idxs_dict,
    idx2filename_dict,
    valid_ratio=0.2,
    min_prompt_num=16,
):
    """
    Split into pool/valid sets and ensure each valid sample has at least
    min_prompt_num available prompts.
    """
    pool_indices_by_fold, valid_indices_by_fold = split_pedestrians_by_ratio(
        filename_list, pedestrians_list, frames_list, valid_ratio=valid_ratio
    )
    pool_indices_by_fold_filtered = []
    valid_indices_by_fold_filtered = []

    for i, (pool_indices, valid_indices) in enumerate(
        zip(pool_indices_by_fold, valid_indices_by_fold)
    ):
        pool_indices_set = set(pool_indices)
        print("*" * 20)
        print(f"fold {i}")
        print("pool data num: " + str(len(pool_indices)))
        print("valid data num: " + str(len(valid_indices)))
        valid_indices = [
            idx
            for idx in valid_indices
            if sum(
                1
                for i in filename2idxs_dict[idx2filename_dict[idx]]
                if i in pool_indices_set and i != idx
            )
            >= min_prompt_num
        ]
        print("valid data num filtered by min_prompt_num: " + str(len(valid_indices)))
        pool_indices_by_fold_filtered.append(pool_indices)
        valid_indices_by_fold_filtered.append(valid_indices)

    return pool_indices_by_fold_filtered, valid_indices_by_fold_filtered


def save_data(
    save_name,
    split,
    trajs,
    masks,
    valid_indices_by_fold,
    pool_indices_by_fold,
    filename2idxs_dict,
    idx2filename_dict,
    config,
    filename_list,
    frames_list,
    pedestrians_list,
    save_root="processed_data",
):
    """
    Save processed dataset under the specified directory.
    """
    print(f"Saving processed data to {save_name}...")
    save_dir = os.path.join(save_root, save_name)
    os.makedirs(save_dir, exist_ok=True)

    torch.save(trajs, os.path.join(save_dir, f"{split}_trajs.pt"))
    torch.save(masks, os.path.join(save_dir, f"{split}_masks.pt"))

    pickle_dump(valid_indices_by_fold, os.path.join(save_dir, f"{split}_valid_indices_by_fold.pickle"))
    pickle_dump(pool_indices_by_fold, os.path.join(save_dir, f"{split}_pool_indices_by_fold.pickle"))
    pickle_dump(filename2idxs_dict, os.path.join(save_dir, f"{split}_filename2idxs_dict.pickle"))
    pickle_dump(idx2filename_dict, os.path.join(save_dir, f"{split}_idx2filename_dict.pickle"))

    with open(os.path.join(save_dir, f"config_{split}.json"), mode="wt") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    pickle_dump(filename_list, os.path.join(save_dir, f"{split}_filename_list.pickle"))
    pickle_dump(frames_list, os.path.join(save_dir, f"{split}_frames_list.pickle"))
    pickle_dump(pedestrians_list, os.path.join(save_dir, f"{split}_pedestrians_list.pickle"))


# ============================================================
#  Step2: Compute and save similarity matrices
# ============================================================

def load_processed_data(
    save_name,
    split,
    save_root="processed_data",
    similarity_scope=None,
):
    """
    Load processed data from processed_data folder.
    If similarity_scope is specified, also load sim_matrix_dicts.
    """
    save_dir = os.path.join(save_root, save_name)
    trajs = torch.load(f"{save_dir}/{split}_trajs.pt")
    masks = torch.load(f"{save_dir}/{split}_masks.pt")
    with open(f"{save_dir}/{split}_filename2idxs_dict.pickle", mode="br") as fi:
        filename2idxs_dict = pickle.load(fi)
    with open(f"{save_dir}/{split}_idx2filename_dict.pickle", mode="br") as fi:
        idx2filename_dict = pickle.load(fi)
    with open(f"{save_dir}/config_{split}.json", mode="br") as fi:
        config = json.load(fi)
    with open(f"{save_dir}/{split}_filename_list.pickle", mode="br") as fi:
        filename_list = pickle.load(fi)
    with open(f"{save_dir}/{split}_frames_list.pickle", mode="br") as fi:
        frames_list = pickle.load(fi)
    with open(f"{save_dir}/{split}_pedestrians_list.pickle", mode="br") as fi:
        pedestrians_list = pickle.load(fi)

    with open(f"{save_dir}/{split}_valid_indices_by_fold.pickle", mode="br") as fi:
        valid_indices_by_fold = pickle.load(fi)

    with open(f"{save_dir}/{split}_pool_indices_by_fold.pickle", mode="br") as fi:
        pool_indices_by_fold = pickle.load(fi)

    sim_matrix_dicts = None
    if similarity_scope is not None:
        sim_matrix_dicts = torch.load(
            os.path.join(save_dir, f"{split}_{similarity_scope}_sim_matrix_dicts.pt")
        )

    return (
        trajs,
        masks,
        filename2idxs_dict,
        idx2filename_dict,
        config,
        filename_list,
        frames_list,
        pedestrians_list,
        valid_indices_by_fold,
        pool_indices_by_fold,
        sim_matrix_dicts,
    )


def compute_sim_matrix(
    trajs,
    filename2idxs_dict,
    hist_len=9,
    save_dir="data/motsynth_loc/",
    split="train",
    load_precomputed=False,
    similarity_scope="hist",
):
    """
    Compute similarity matrices for trajectories (distance & velocity).
    """

    sim_matrix_dicts = {}
    sim_items = ["dist", "vel"]

    if load_precomputed:
        print("Loading similarity matrix from file...")
        sim_matrix_dicts = torch.load(
            os.path.join(save_dir, f"{split}_{similarity_scope}_sim_matrix_dicts.pt")
        )
        print("Finished loading similarity matrix.")
    else:
        for item in sim_items:
            sim_matrix_dicts[item] = {}

        def process_file(filename, idxs):
            if similarity_scope == "hist":
                primary_trajs = [trajs[idx][0, :hist_len, 0, :2] for idx in idxs]
            elif similarity_scope == "seq":
                primary_trajs = [trajs[idx][0, :, 0, :2] for idx in idxs]

            primary_trajs = torch.stack(primary_trajs)
            primary_trajs_target = primary_trajs

            dist_matrix = torch.cdist(
                primary_trajs.view(len(idxs), -1),
                primary_trajs_target.view(len(idxs), -1),
            )
            velocities = primary_trajs[:, 1:] - primary_trajs[:, :-1]
            velocities_target = primary_trajs_target[:, 1:] - primary_trajs_target[:, :-1]
            vel_matrix = torch.cdist(
                velocities.view(len(idxs), -1),
                velocities_target.view(len(idxs), -1),
            )

            dist_matrix = (dist_matrix - dist_matrix.min()) / (dist_matrix.max() - dist_matrix.min() + 1e-8)
            vel_matrix = (vel_matrix - vel_matrix.min()) / (vel_matrix.max() - vel_matrix.min() + 1e-8)

            torch.cuda.empty_cache()

            return {"filename": filename, "dist": dist_matrix, "vel": vel_matrix}

        print("Computing similarity matrix...")
        results = []
        for filename, idxs in tqdm(filename2idxs_dict.items(), desc="Processing files"):
            result = process_file(filename, idxs)
            results.append(result)
        print("Finished computing similarity matrix.")

        for result in results:
            sim_matrix_dicts["dist"][result["filename"]] = result["dist"]
            sim_matrix_dicts["vel"][result["filename"]] = result["vel"]

        out_path = os.path.join(save_dir, f"{split}_{similarity_scope}_sim_matrix_dicts.pt")
        print(out_path)
        torch.save(sim_matrix_dicts, out_path)

    return sim_matrix_dicts


# ============================================================
#  Step3: Compute trajectory similarity using sim_matrix
# ============================================================

def process_file_optimized(args):
    """
    Optimized file processing function for parallel execution.
    """
    (filename, idxs, sim_matrix_dicts, dist_weight, vel_weight, threshold, max_similar) = args

    dist_matrix = sim_matrix_dicts["dist"][filename]
    vel_matrix = sim_matrix_dicts["vel"][filename]

    combined_similarity = dist_weight * dist_matrix + vel_weight * vel_matrix

    if isinstance(combined_similarity, torch.Tensor):
        combined_similarity = combined_similarity.clone()
        combined_similarity.fill_diagonal_(float("inf"))
    else:
        np.fill_diagonal(combined_similarity, float("inf"))

    similarity_scores = 1 / (1 + combined_similarity)

    file_similar_trajs = {}
    file_similarity_scores = {}

    for i, idx in enumerate(idxs):
        scores_i = similarity_scores[i]

        if isinstance(scores_i, torch.Tensor):
            valid_mask = scores_i > threshold
            valid_indices = torch.where(valid_mask)[0].cpu().numpy()
        else:
            valid_mask = scores_i > threshold
            valid_indices = np.where(valid_mask)[0]

        if max_similar is None:
            if isinstance(scores_i, torch.Tensor):
                scores_valid = scores_i[valid_indices]
                sorted_idx = torch.argsort(scores_valid, descending=True).cpu().numpy()
                sorted_indices = valid_indices[sorted_idx]
                sorted_scores = scores_valid[sorted_idx].cpu().numpy().tolist()
            else:
                sorted_indices = valid_indices[np.argsort(scores_i[valid_indices])[::-1]]
                sorted_scores = scores_i[sorted_indices].tolist()

            similar_indices = [idxs[j] for j in sorted_indices]

        else:
            if len(valid_indices) < max_similar:
                if isinstance(scores_i, torch.Tensor):
                    sorted_indices = torch.argsort(scores_i, descending=True).cpu().numpy()
                    similar_indices = [idxs[j] for j in sorted_indices if j != i][:max_similar]
                    sorted_scores = scores_i[sorted_indices[sorted_indices != i]][:max_similar].cpu().numpy().tolist()
                else:
                    sorted_indices = np.argsort(scores_i)[::-1]
                    similar_indices = [idxs[j] for j in sorted_indices if j != i][:max_similar]
                    sorted_scores = scores_i[sorted_indices[sorted_indices != i]][:max_similar].tolist()
            else:
                if isinstance(scores_i, torch.Tensor):
                    scores_valid = scores_i[valid_indices]
                    sorted_idx = torch.argsort(scores_valid, descending=True).cpu().numpy()
                    sorted_indices = valid_indices[sorted_idx]
                    sorted_scores = scores_valid[sorted_idx].cpu().numpy().tolist()
                else:
                    sorted_indices = valid_indices[np.argsort(scores_i[valid_indices])[::-1]]
                    sorted_scores = scores_i[sorted_indices].tolist()

                similar_indices = [idxs[j] for j in sorted_indices]

        file_similar_trajs[idx] = similar_indices
        file_similarity_scores[idx] = sorted_scores[:16]

    return file_similar_trajs, file_similarity_scores


def compute_trajectory_similarity(
    filename2idxs_dict,
    dist_weight=1,
    vel_weight=1,
    threshold=0,
    max_similar=None,
    pool_indices=None,
    sim_matrix_dicts=None,
    use_parallel=True,
    max_workers=None,
):
    """
    Compute trajectory similarity scores for all trajectories.
    """

    similar_trajs = {}
    similar_scores = {}

    if max_workers is None:
        max_workers = min(cpu_count(), len(filename2idxs_dict))

    print("Calculating similarity...")

    if use_parallel and len(filename2idxs_dict) > 1:
        args_list = [
            (filename, idxs, sim_matrix_dicts, dist_weight, vel_weight, threshold, max_similar)
            for filename, idxs in filename2idxs_dict.items()
        ]

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_file_optimized, args) for args in args_list]

            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
                result_sim, result_scores = future.result()
                similar_trajs.update(result_sim)
                similar_scores.update(result_scores)
    else:
        for filename, idxs in tqdm(filename2idxs_dict.items(), desc="Processing files"):
            args = (filename, idxs, sim_matrix_dicts, dist_weight, vel_weight, threshold, max_similar)
            result_sim, result_scores = process_file_optimized(args)
            similar_trajs.update(result_sim)
            similar_scores.update(result_scores)

    print("Finished calculating similarity.")

    if pool_indices is not None:
        pool_indices_set = set(pool_indices)
        similar_trajs = {
            key: [idx for idx in value if idx in pool_indices_set]
            for key, value in similar_trajs.items()
        }
    print("finish computing similarity")

    return similar_trajs, similar_scores


# ============================================================
#  Automatic r, stride setting
# ============================================================

def infer_r_stride(name):
    """
    Extracted from original if/elif.
    """
    stride = 21
    if name == "motsynth":
        r = 50
    elif name == "jrdb":
        r = 6
    elif name == "jta":
        r = 2
    return r, stride


# ============================================================
#  Main: Argument-based pipeline controller
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Unified pipeline: preprocess -> sim_matrix -> traj_similarity"
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["preprocess", "sim_matrix", "traj_sim", "all"],
        help="Which stage(s) to run",
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Dataset name (e.g., motsynth, jrdb, jta)",
    )
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--save_root", type=str, default="processed_data")
    parser.add_argument(
        "--splits",
        type=str,
        default="train,val",
        help="Comma-separated dataset splits",
    )
    parser.add_argument(
        "--similarity_scopes",
        type=str,
        default="hist,seq",
        help="Similarity scopes (e.g., hist,seq)",
    )
    parser.add_argument("--valid_ratio", type=float, default=0.2)
    parser.add_argument("--min_prompt_num", type=int, default=16)
    parser.add_argument("--hist_len", type=int, default=9)

    parser.add_argument("--dist_weight", type=float, default=1.0)
    parser.add_argument("--vel_weight", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument(
        "--max_similar",
        type=int,
        default=16,
        help="Maximum number of similar trajectories per sample",
    )
    parser.add_argument("--no_parallel", action="store_true")
    parser.add_argument("--max_workers", type=int, default=None)
    parser.add_argument("--load_precomputed", action="store_true")

    args = parser.parse_args()

    splits = [s.strip() for s in args.splits.split(",")]
    similarity_scopes = [s.strip() for s in args.similarity_scopes.split(",")]

    r, stride = infer_r_stride(args.name)
    save_name = args.name

    # -----------------------------
    # Stage 1: Data preprocessing
    # -----------------------------
    if args.stage in ["preprocess", "all"]:
        print("===== Stage 1: preprocess (load_data -> pool/valid split -> save) =====")
        for split in splits:
            print(f"[Preprocess] name={args.name}, split={split}")

            (
                trajs,
                masks,
                filename_list,
                frames_list,
                pedestrians_list,
                filename2idxs_dict,
                idx2filename_dict,
                config,
            ) = load_data(split, args.name, args.data_dir, r, stride)

            pool_indices_by_fold, valid_indices_by_fold = pool_valid_split(
                filename_list,
                frames_list,
                pedestrians_list,
                filename2idxs_dict,
                idx2filename_dict,
                valid_ratio=args.valid_ratio,
                min_prompt_num=args.min_prompt_num,
            )

            save_data(
                save_name,
                split,
                trajs,
                masks,
                valid_indices_by_fold,
                pool_indices_by_fold,
                filename2idxs_dict,
                idx2filename_dict,
                config,
                filename_list,
                frames_list,
                pedestrians_list,
                save_root=args.save_root,
            )

    # -----------------------------
    # Stage 2: Similarity matrix computation
    # -----------------------------
    if args.stage in ["sim_matrix", "all"]:
        print("===== Stage 2: compute sim_matrix =====")
        for split in splits:
            print(f"[SimMatrix] save_name={save_name}, split={split}")

            (
                trajs,
                masks,
                filename2idxs_dict,
                idx2filename_dict,
                config,
                filename_list,
                frames_list,
                pedestrians_list,
                valid_indices_by_fold,
                pool_indices_by_fold,
                _,
            ) = load_processed_data(save_name, split, args.save_root)

            save_dir = os.path.join(args.save_root, save_name)

            for similarity_scope in similarity_scopes:
                print(f"  -> similarity_scope={similarity_scope}")
                compute_sim_matrix(
                    trajs,
                    filename2idxs_dict,
                    hist_len=args.hist_len,
                    save_dir=save_dir,
                    split=split,
                    load_precomputed=args.load_precomputed,
                    similarity_scope=similarity_scope,
                )

    # -----------------------------
    # Stage 3: Build similar trajectory dictionaries
    # -----------------------------
    if args.stage in ["traj_sim", "all"]:
        print("===== Stage 3: compute trajectory similarity dicts =====")
        for split in splits:
            for similarity_scope in similarity_scopes:
                print(f"[TrajSim] save_name={save_name}, split={split}, scope={similarity_scope}")

                save_dir = os.path.join(args.save_root, save_name)

                (
                    trajs,
                    masks,
                    filename2idxs_dict,
                    idx2filename_dict,
                    config,
                    filename_list,
                    frames_list,
                    pedestrians_list,
                    valid_indices_by_fold,
                    pool_indices_by_fold,
                    sim_matrix_dicts,
                ) = load_processed_data(
                    save_name,
                    split,
                    args.save_root,
                    similarity_scope=similarity_scope,
                )

                similar_traj_dicts = []
                similar_scores_dicts = []

                for i, (valid_indices, pool_indices) in enumerate(
                    zip(valid_indices_by_fold, pool_indices_by_fold)
                ):
                    print(f"  Fold {i}: pool={len(pool_indices)}, valid={len(valid_indices)}")

                    similar_traj_dict, similar_scores_dict = compute_trajectory_similarity(
                        filename2idxs_dict,
                        dist_weight=args.dist_weight,
                        vel_weight=args.vel_weight,
                        threshold=args.threshold,
                        max_similar=args.max_similar,
                        pool_indices=pool_indices,
                        sim_matrix_dicts=sim_matrix_dicts,
                        use_parallel=not args.no_parallel,
                        max_workers=args.max_workers,
                    )

                    similar_traj_dicts.append(similar_traj_dict)
                    similar_scores_dicts.append(similar_scores_dict)

                out_path = os.path.join(
                    save_dir,
                    f"{split}_similar_traj_dicts_{similarity_scope}.pickle",
                )
                pickle_dump(similar_traj_dicts, out_path)
                print(f"  -> saved similar_traj_dicts to {out_path}")


if __name__ == "__main__":
    main()
