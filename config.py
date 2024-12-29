from sfs.model.autoencoders import AutoEncoderCfg
from sfs.model.render_pipelines import RenderPipelineCfg
from sfs.model.pl_wrapper import OptimizerCfg, TrainCfg, TestCfg
from sfs.model.loss import LossCfgWrapper
from sfs.datasets.data_module import DataLoaderCfg, DatasetCfg

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Union, Type, TypeVar, List, Tuple

from dacite import Config, from_dict
from omegaconf import DictConfig, OmegaConf


@dataclass
class CheckpointingCfg:
    load: Optional[str]  # Not a path, since it could be something like wandb://...
    every_n_train_steps: int
    save_top_k: int
    resume: Optional[bool] = True

@dataclass
class ModelCfg:
    shape: AutoEncoderCfg
    render: RenderPipelineCfg


@dataclass
class TrainerCfg:
    max_steps: int
    val_check_interval: Union[int , float, None]
    gradient_clip_val: Union[int , float , None]
    num_sanity_val_steps: int
    num_nodes: Optional[int] = 1


@dataclass
class RootCfg:
    wandb: dict
    mode: Literal["train", "test"]
    dataset: DatasetCfg
    data_loader: DataLoaderCfg
    model: ModelCfg
    optimizer: OptimizerCfg
    checkpointing: CheckpointingCfg
    loss: list[LossCfgWrapper]
    trainer: TrainerCfg
    train: TrainCfg
    test: TestCfg
    seed: int


TYPE_HOOKS = {
    Path: Path,
    Tuple: lambda x: tuple(x)
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


def separate_loss_cfg_wrappers(joined: dict) -> list[LossCfgWrapper]:
    # The dummy allows the union to be converted.
    @dataclass
    class Dummy:
        dummy: LossCfgWrapper

    return [
        load_typed_config(DictConfig({"dummy": {k: v}}), Dummy).dummy
        for k, v in joined.items()
    ]


def load_typed_root_config(cfg: DictConfig) -> RootCfg:
    return load_typed_config(
        cfg,
        RootCfg,
        {list[LossCfgWrapper]: separate_loss_cfg_wrappers},
    )
