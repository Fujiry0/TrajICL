import argparse
import warnings
from typing import Any, Dict

import torch
from omegaconf import OmegaConf

from helper import create_dataloader, evaluate
from model import create_model

# Ignore minor warnings to keep the output clean
warnings.simplefilter("ignore")

def parse_args():
    """
    Parses command line arguments for the evaluation script.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    # --- 🎯 Argument Parser Setup ---
    parser = argparse.ArgumentParser(description="Evaluate a trajectory prediction model.")
    parser.add_argument(
        "--model_path",
        type=str,
        default="results/TrajICL/worthy-firebrand-50/best_val_checkpoint.pth.tar",
        help="Path to the model checkpoint file.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="motsynth",
        help="Name of the dataset to evaluate on.",
    )
    parser.add_argument(
        "--prompting_method",
        type=str,
        default="sim",
        help="Prompting method setting for the dataset configuration.",
    ) # random / sim
    return parser.parse_args()

def main():
    """
    Main function to load the model, run evaluation across various shot counts,
    and print the results.
    """
    args = parse_args()

    # --- 🔧 Apply Parameters (Setup variables used in existing logic) ---
    device = "cuda"

    model_path = args.model_path
    print(f"Loading model from: {model_path}")
    dataset_name = args.dataset_name
    # ----------------------------------------------------

    print(f"Dataset name: {dataset_name}")

    # Extract project/run name from the model path
    run_name = model_path.split("/")[-2]

    # --- ⚙️ Load Checkpoint and Configuration ---
    # The device is set to 'cuda' as per the original script
    checkpoint = torch.load(
        model_path,
        map_location=device,
    )
    # The configuration is stored under the 'cfg' key
    cfg = checkpoint["cfg"]

    # Ensure cfg is mutable and disable structure enforcement for modification
    cfg = OmegaConf.create(cfg)
    OmegaConf.set_struct(cfg, False)

    # --- 🔄 Update Configuration with Arguments ---
    cfg.dataset.prompting = args.prompting_method
    cfg.dataset.name = dataset_name # Update dataset name for dataloader creation


    # --- 🏗️ Model Initialization and Loading ---
    model = create_model(cfg)
    model.load_state_dict(checkpoint["model"])
    model = model.to(device)
    model.eval() # Set model to evaluation mode

    # Determine the dataset split for evaluation
    split = "val" # Use 'val' split for 'motsynth' dataset

    # Dictionaries to store results for ADE and FDE vs. shot count
    ade_vs_shot_results: Dict[int, float] = {}
    fde_vs_shot_results: Dict[int, float] = {}

    # --- ループ 🚀 Evaluation Loop over Shot Counts ---
    # Evaluate for various numbers of demonstration examples (shots)
    for i in [0, 2, 4, 8]:
        print(f"--- Running evaluation for {i} shots ---")
        cfg.dataset.num_example = i  # i corresponds to the number of shots (x-axis)

        # Create the DataLoader for the current configuration
        dataloader_test = create_dataloader(
            split=split, dataset_name=cfg.dataset.name, cfg=cfg
        )

        stats: Dict[str, Any] = {}
        # Run the evaluation
        stats = evaluate(
            "test",
            cfg,
            0,
            model,
            dataloader_test,
            stats,
        )

        # --- Scale Results by the Resize Factor ---
        # Divide the results by the resize factor (existing processing logic)
        for key, val in stats.items():
            # Check if the value is a number before division
            if isinstance(val, (int, float)):
                stats[key] = val / cfg.training.resize

        # --- Save Results ---
        # Get ADE (Average Displacement Error) and FDE (Final Displacement Error)
        # results and store them associated with the current shot count 'i'.
        # Assumes 'loss_ade/test' and 'loss_fde/test' are keys in the stats dictionary.
        ade_vs_shot_results[i] = stats["loss_ade/test"]
        fde_vs_shot_results[i] = stats["loss_fde/test"]

    # --- 📈 Final Results Formatting and Output ---
    print("\n" + "="*50)
    print(f"✅ Final Results: **{run_name}** on **{cfg.dataset.name}**")
    print("="*50)

    print("### 📊 minADE&minFDE vs Shot Summary")

    # Table format output: Shot (x-axis) vs. ADE/FDE (y-axis)
    print("\n| Shot (num_example) | minADE | minFDE |")
    # Ensure the column alignment for table
    print("| :----------------: | :----: | :----: |")

    # Display sorted by shot count (i)
    for (shot, ade), (_, fde) in sorted(zip(ade_vs_shot_results.items(), fde_vs_shot_results.items())):
        print(f"| {shot:<18} | {ade:.4f} | {fde:.4f} |")

    print("="*50)

if __name__ == "__main__":
    main()
