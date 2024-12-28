from model.autoencoders import AutoEncoderCfg

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Type, TypeVar, Any

from dacite import Config, from_dict
from omegaconf import DictConfig, OmegaConf


@dataclass
class CheckpointingCfg:
    load: Optional[str]  # Not a path, since it could be something like wandb://...
    every_n_train_steps: int
    save_top_k: int
    pretrained_model: Optional[str]
    resume: Optional[bool] = True




@dataclass
class TrainerCfg:
    max_steps: int
    val_check_interval: int | float | None
    gradient_clip_val: int | float | None
    num_sanity_val_steps: int
    num_nodes: Optional[int] = 1


@dataclass
class RootCfg:
    wandb: dict
    mode: Literal["train", "test"]
    shape: AutoEncoderCfg
    render: Any
    checkpointing: CheckpointingCfg
    trainer: TrainerCfg
    seed: int


TYPE_HOOKS = {
    Path: Path,
}


T = TypeVar("T")


def load_typed_config(
    cfg: DictConfig,
    data_class: Type[T],
    extra_type_hooks: dict = {},
) -> T:
    return from_dict(
        data_class,
        OmegaConf.to_container(cfg),
        config=Config(type_hooks={**TYPE_HOOKS, **extra_type_hooks}),
    )



def load_typed_root_config(cfg: DictConfig) -> RootCfg:
    return load_typed_config(
        cfg,
        RootCfg,
    )