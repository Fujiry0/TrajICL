<div align="center">

# Towards Predicting Any Human Trajectory In Context (NeurIPS 2025)

**[[Paper](https://arxiv.org/abs/2506.00871)] [[Project Page](https://fujiry0.github.io/TrajICL-project-page/)] [[Poster](https://neurips.cc/media/PosterPDFs/NeurIPS%202025/115894.png?t=1763031083.2285979)]**

</div>

This is the official code release for our NeurIPS 2025 paper "Towards Predicting Any Human Trajectory In Context".

## 🔍 TrajICL

![TrajICL](./misc/concept.png)

Predicting accurate future trajectories requires adaptability, yet fine-tuning for each new scenario is often impractical for edge deployment. To address this, we introduce TrajICL, an In-Context Learning (ICL) framework for pedestrian trajectory prediction that enables robust adaptation to diverse environments at inference time without requiring weight updates.

Our TrajICL implementation includes the following key features:

- **Spatio-Temporal Similarity-based Example Selection (STES):** Selects relevant examples from observed trajectories by identifying similar motion patterns at corresponding locations within the same scene.
- **Prediction-Guided Example Selection (PG-ES):** Refines example selection by utilizing both past and predicted future trajectories to account for long-term dynamics.
- **Superior Adaptation & Generalization:** Leverages large-scale synthetic training to achieve remarkable adaptation, outperforming even fine-tuned approaches across in-domain and cross-domain benchmarks.

## 📂 Data Preparation

### Download MOTSynth dataset

```bash
bash bash scripts/donwnload.sh
```

### Run Preprocessing Code

```bash
bash bash scripts/preprocess.sh
```

## 🔥 Training

### 1. Vanilla trajectory prediction (VTP) training

VTP checkpoints are saved in `results/TrajICL`.

```bash
python train.py
```

### 2. In-context training

For load_model.model_path, please specify the relative path (inside the `results/TrajICL` directory) to the checkpoint saved during VTP training.

Example: If the full path is `results/TrajICL/robust-sunset-33/best_val_checkpoint.pth`.tar, you should set load_model.model_path to `robust-sunset-33/best_val_checkpoint.pth`.tar.

```bash
python train.py -m training.epochs=400 training.warmup_steps=12 dataset.num_example=8　load_model.model_path=robust-sunset-33/best_val_checkpoint.pth.tar
```

> ⚠️ Note: The path `robust-sunset-33/best_val_checkpoint.pth.tar` shown in the -m option above is just an example. Please modify this value to match the actual checkpoint path generated after running the VTP training (Step 1).

## 🔍 Evaluation

```bash
python eval.py
```

## ✅ TODO

- [ ] Add prediction-guided example selection code
- [ ] Add other datasets

## 👏 Acknowledgement

We sincerely thank the authors of Social-Transmotion for providing their [source code](https://github.com/vita-epfl/social-transmotion), which has been invaluable to our work. We are immensely grateful for their contribution.

## ✍️ Citation

If you use this code for your research, please cite our paper.

```bib
@article{Fujii2025TrajICL,
  title = {Towards Predicting Any Human Trajectory In Context},
  author= {Fujii, Ryo and Hachiuma, Ryo and Saito, Hideo},
  journal={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2025}
}
```
