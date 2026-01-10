import warnings
from collections import defaultdict

import hydra
from omegaconf import DictConfig
from torch.optim import Adam, AdamW
from transformers import get_cosine_schedule_with_warmup

from helper import (
    adjust_learning_rate,
    evaluate,
    evaluate_and_update_min_val,
    prepare_dataloader,
    set_seed,
    train,
)
from model import create_model
from utils.utils import (
    freeze_params,
    get_nb_trainable_parameters,
    load_model_checkpoint,
    logging_wandb,
    setup_wandb_logging,
)

warnings.simplefilter("ignore")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):

    set_seed(cfg.training.seed)
    wandb, output_dir = setup_wandb_logging(cfg)
    dataloader_train, dataloader_val = prepare_dataloader(cfg)

    model = create_model(cfg)

    if cfg.training.optimizer == "adamw":
        optimizer = AdamW(
            model.parameters(),
            lr=cfg["training"]["lr"],
            weight_decay=cfg["training"]["weight_decay"],
        )
    elif cfg.training.optimizer == "adam":
        optimizer = Adam(
            model.parameters(),
            lr=cfg["training"]["lr"],
            weight_decay=cfg["training"]["weight_decay"],
        )

    scheduler = None
    if cfg.training.scheduler == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=cfg["training"]["warmup_steps"],
            num_training_steps=cfg.training.epochs * len(dataloader_train),
        )
    start_epoch = 0
    if cfg.load_model.model_path:
        start_epoch, model, optimizer, scheduler = load_model_checkpoint(
            cfg, model, optimizer, scheduler
        )

    if cfg.training.freeze_encoder:
        freeze_params(model)

    trainable_params, all_param = get_nb_trainable_parameters(model)
    print(
        f"trainable params: {trainable_params:,d} || "
        f"all params: {all_param:,d} || "
        f"trainable%: {100 * trainable_params / all_param:.4f}"
    )
    min_val = defaultdict(lambda: 1e4)
    for epoch in range(start_epoch, cfg.training.epochs):
        if cfg.training.scheduler == "adjust_lr":
            adjust_learning_rate(optimizer, epoch, cfg)
        stats = {}
        stats = train(
            cfg,
            epoch,
            dataloader_train,
            model,
            optimizer,
            scheduler,
            stats,
        )
        stats = evaluate(
            "val",
            cfg,
            epoch,
            model,
            dataloader_val,
            stats,
        )
        stats, min_val = evaluate_and_update_min_val(
            cfg,
            epoch,
            model,
            stats,
            min_val,
            output_dir,
            optimizer,
            scheduler,
        )
        logging_wandb(cfg, model, optimizer, scheduler, epoch, stats, output_dir, wandb)
        print("epoch ", epoch, " finished!")

    if cfg.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
