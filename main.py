# Copyright (c) 2024 Yang Li, Microsoft. All rights reserved.
# MIT License.
# 
# --------------------------------------------------------
# main.py
# --------------------------------------------------------
# 
# Created on Thu Dec 26 2024.


import os
import sys
from pathlib import Path
import warnings
import shutil

import hydra
import torch
import wandb
from colorama import Fore
from jaxtyping import install_import_hook
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers.wandb import WandbLogger


# Configure beartype and jaxtyping.
with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from config import load_typed_root_config
    from global_config import set_cfg
    from sfs.datasets.data_module import DataModule
    from sfs.model.loss import get_losses
    from sfs.model.autoencoders import get_autoencoder
    from sfs.model.render_pipelines import get_renderer
    from sfs.model.pl_wrapper import LightningWrapper
    from sfs.utils.logger import std_logger, cyan

# Backup Code Callback
class SaveCodeCallback(Callback):
    def __init__(self, src_dir, dest_dir):
        self.src_dir = src_dir
        self.dest_dir = dest_dir

    def on_train_start(self, trainer, pl_module):
        os.makedirs(self.dest_dir, exist_ok=True)
        for root, dirs, files in os.walk(self.src_dir):
            dest_root = os.path.join(self.dest_dir, os.path.relpath(root, self.src_dir))
            if not os.path.exists(dest_root):
                os.makedirs(dest_root)
            for file in files:
                if file.endswith('.py'):
                    src_file = os.path.join(root, file)
                    dest_file = os.path.join(dest_root, file)
                    shutil.copy2(src_file, dest_file)
        if isinstance(trainer.logger, WandbLogger):
            trainer.logger.experiment.config.update({"code_backup_dir": self.dest_dir})
        std_logger.info(f'Code backup into '+ cyan(str(self.dest_dir)))


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="main.yaml",
)
def train(cfg_dict: DictConfig):
    breakpoint()
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)
    breakpoint()
    # Set up the output directory.
    if cfg_dict.output_dir is None:
        output_dir = Path(cfg_dict.root_dir) / Path(
            hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
        )
    else:  # for resuming
        output_dir = Path(cfg_dict.root_dir) / Path(cfg_dict.output_dir)
        os.makedirs(output_dir, exist_ok=True)
    std_logger.info(cyan(f"Saving outputs to {output_dir}."))
    latest_run = output_dir.parents[1] / "latest-run"
    os.system(f"rm {latest_run}")
    os.system(f"ln -s {output_dir} {latest_run}")

    # Set up logging with wandb.
    callbacks = []
    if cfg_dict.wandb.mode != "disabled" and cfg.mode == 'train':
        wandb_extra_kwargs = {}
        if cfg_dict.wandb.id is not None:
            wandb_extra_kwargs.update({'id': cfg_dict.wandb.id,
                                       'resume': "must"})
        logger = WandbLogger(
            entity=cfg_dict.wandb.entity,
            project=cfg_dict.wandb.project,
            mode=cfg_dict.wandb.mode,
            name=f"{cfg_dict.wandb.name} ({output_dir.parent.name}/{output_dir.name})",
            tags=cfg_dict.wandb.get("tags", None),
            log_model=False,
            save_dir=output_dir,
            config=OmegaConf.to_container(cfg_dict),
            **wandb_extra_kwargs,
        )
        callbacks.append(LearningRateMonitor("step", True))

        # On rank != 0, wandb.run is None.
        if wandb.run is not None:
            wandb.run.log_code("model")


    # Set up checkpointing.
    callbacks.append(
        ModelCheckpoint(
            output_dir / "checkpoints",
            every_n_train_steps=cfg.checkpointing.every_n_train_steps,
            save_top_k=cfg.checkpointing.save_top_k,
            monitor="info/global_step",
            mode="max",  # save the lastest k ckpt, can do offline test later
        ),
    )
    
    # Backup Src Code
    callbacks.append(SaveCodeCallback(
            'model',
            output_dir / "code"
        ))


    trainer = Trainer(
        max_epochs=-1,
        accelerator="gpu",
        logger=logger,
        devices="auto",
        strategy="ddp" if torch.cuda.device_count() > 1 else "auto",
        callbacks=callbacks,
        val_check_interval=cfg.trainer.val_check_interval,
        enable_progress_bar= False,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        max_steps=cfg.trainer.max_steps,
        num_sanity_val_steps=cfg.trainer.num_sanity_val_steps,
    )
    torch.manual_seed(cfg_dict.seed + trainer.global_rank)

    shape_autoencoder = get_autoencoder(cfg.model.shape)
    losses = []
    for loss_cfg in cfg.loss:
        losses.append(get_losses(loss_cfg))

    model_kwargs = {
        "optimizer_cfg": cfg.optimizer,
        "train_cfg": cfg.train,
        "shape_ae": shape_autoencoder,
        "render_pipeline": get_renderer(cfg.model.render),
        "losses":losses,
    }

    checkpoint_path = cfg.checkpointing.load
    
    if cfg.mode == "train" and checkpoint_path is not None and not cfg.checkpointing.resume:
        # Just load model weights, without optimizer states
        # e.g., fine-tune from the released weights on other datasets
        model_wrapper = LightningWrapper.load_from_checkpoint(
            checkpoint_path, **model_kwargs, strict=True)
        std_logger.info(cyan(f" Loaded weigths from {checkpoint_path}."))
    elif checkpoint_path is not None:
        model_wrapper = LightningWrapper.load_from_checkpoint(
            checkpoint_path, **model_kwargs, strict=False)
        
    else:
        model_wrapper = LightningWrapper(**model_kwargs)

    data_module = DataModule(
        cfg.dataset,
        cfg.data_loader,
        global_rank=trainer.global_rank,
    )

    if cfg.mode == "train":
        trainer.fit(model_wrapper, datamodule=data_module, ckpt_path=(
            checkpoint_path if cfg.checkpointing.resume else None))




if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    torch.set_float32_matmul_precision('high')

    train()
