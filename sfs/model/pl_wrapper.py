# Copyright (c) 2024 Yang Li, Microsoft. All rights reserved.
# MIT License.
# 
# --------------------------------------------------------
# Pytorch Lightning Module Wrapper for Shape-from-shadow
# --------------------------------------------------------
# 
# Created on Wed Dec 25 2024.


from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Literal, List, Dict

import torch
from torch import nn
import wandb
from einops import pack, rearrange, repeat
from jaxtyping import Float
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities import rank_zero_only

from .autoencoders import ImplementedAutoEncoders
from .render_pipelines import ImplementedRenderPipelines
from .loss import Loss
from .types import ReconCues, OptimizationVariables

from ..utils.metrics import compute_IoU
from ..utils.logger import std_logger, cyan, yellow
from ..utils.sys_monitor import Benchmarker
from ..utils.visualization import visualize_pred_gt_comparison
from ..utils.export import export_pts

from global_config import get_cfg

@dataclass
class OptimizerCfg:
    lr: float
    warm_up_steps: int
    cosine_lr: bool
    opt_params: List[Literal["light_position", "latent_set", "object_pose", "object_scale"]]
    lr_decay: Dict[Literal["light_position", "latent_set", "object_pose", "object_scale"], float]
    

@dataclass
class TrainCfg:
    log_every_n_steps: int = 100
    print_log_every_n_steps: int = 1
    output_path: Path = Path('outputs/local')
    
@dataclass
class TestCfg:
    placeholder: str = 'placeholder'


class LitWrapper(LightningModule):
    logger: Optional[WandbLogger]
    shape_ae: ImplementedAutoEncoders
    render_pipeline: ImplementedRenderPipelines
    losses: List[Loss]
    optimizer_cfg: OptimizerCfg
    train_cfg: TrainCfg
    benchmarker: Benchmarker

    def __init__(
        self,
        optimizer_cfg: OptimizerCfg,
        train_cfg: TrainCfg,
        shape_ae: ImplementedAutoEncoders,
        render_pipeline: ImplementedRenderPipelines,
        losses: List[Loss],
    ) -> None:
        super().__init__()


        self.optimizer_cfg = optimizer_cfg
        self.train_cfg = train_cfg

        # Set up the model.
        self.shape_ae = shape_ae
        self.render_pipeline = render_pipeline
        self.losses = losses
        
        # Set up learnable Parameter
        self.scene_context = nn.ParameterDict({
            "light_position": nn.Parameter(torch.tensor([[0.,0.,2.]])),
            "latent_set": nn.Parameter(torch.randn(1,512,8)),
            "object_pose": nn.Parameter(torch.tensor([[0.,0.,0.,1.,0.,0.,0.]],dtype=torch.float32)), # Init w/ no rot & trans
            "object_scale": nn.Parameter(torch.ones(1,1,dtype=torch.float32))
        })
        
        # Set up system benchmarker
        self.benchmarker = Benchmarker()
        
        # Set up outpath
        self.output_path:Path = Path(get_cfg().root_dir) / get_cfg().output_dir



    def training_step(self, batch:ReconCues, batch_idx):

        
        gt_shadow_map = batch["shadow_map"]
        
        # Pred shadow map
        # Generate Occ Volume
        occ_vol = self.shape_ae(self.scene_context)

        # Render Top-view Shadow Map
        pred = self.render_pipeline(self.scene_context, occ_vol)
        
        pred_shadow_map = pred["shadow_map"] # [b, h, w]
    

        self.log("train/IoU", compute_IoU(pred_shadow_map, gt_shadow_map))

        # Compute and log loss.
        total_loss = 0
        for loss_fn in self.losses:
            loss = loss_fn.forward(pred_shadow_map, gt_shadow_map)
            self.log(f"loss/{loss_fn.cfg.name}", loss)
            total_loss = total_loss + loss
        self.log("loss/total", total_loss)


        if (
            self.global_rank == 0
            and self.global_step % self.train_cfg.print_log_every_n_steps == 0
        ):
            std_logger.info(
                f" train step {cyan(self.global_step)}; "
                f"loss = {total_loss:.6f}"
            )

        self.log("info/global_step", self.global_step)  # hack for ckpt monitor


        return total_loss 
    
    @rank_zero_only
    def validation_step(self, batch, batch_idx):
        
        gt_shadow_map = batch["shadow_map"]
        
        if self.global_rank == 0:
            std_logger.info(
                f"{yellow('validation')} step {self.global_step}; "
            )
        
        with self.benchmarker.track_peak_memory("forward"):
            with self.benchmarker.time("3D VAE"):
                # Pred shadow map
                # Generate Occ Volume
                occ_vol = self.shape_ae(self.scene_context)
            with self.benchmarker.time("Render"):
                # Render shadow map
                pred = self.render_pipeline(self.scene_context, occ_vol)
        
        pred_shadow_map = pred["shadow_map"] # [1, h, w]

        # Compute validation metrics.
        self.log("val/IoU", compute_IoU(pred_shadow_map, gt_shadow_map))

        # Construct comparison image.
        comparison = visualize_pred_gt_comparison(pred_shadow_map, gt_shadow_map)
        self.logger.log_image(
            "comparison",
            [comparison],
            step=self.global_step,
            caption=["rendered shadow map v.s. gt shadow map"],
        )
        
        # Export Occ Volume
        pts3d = occ_vol.grid[occ_vol.occ_logits > 0]
        self.output_path.mkdir(parents=True, exist_ok=True)
        export_pts(pts3d, self.output_path / f'occ_step_{self.global_step}.ply')

    def configure_optimizers(self):
        params = []
        for param_name in self.optimizer_cfg.opt_params:
            assert param_name in self.scene_context, f"{param_name} is not a learnable parameter."
            params.append({
                "params": self.scene_context[param_name],
                "lr": self.optimizer_cfg.lr * self.optimizer_cfg.lr_decay[param_name]
            })
        
        optimizer = torch.optim.Adam(params, lr=self.optimizer_cfg.lr)
        warm_up = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.optimizer_cfg.lr,
            total_steps=self.trainer.max_steps + 10,
            pct_start=0.01,
            cycle_momentum=False,
            anneal_strategy='cos',
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": warm_up,
                "interval": "step",
                "frequency": 1,
            },
        }
