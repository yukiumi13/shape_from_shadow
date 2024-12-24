from typing import Callable, Literal, TypedDict

from jaxtyping import Float, Int64
from torch import Tensor
from dataclasses import dataclass



# The following types mainly exist to make type-hinted keys show up in VS Code. Some
# dimensions are annotated as "_" because either:
# 1. They're expected to change as part of a function call (e.g., resizing the dataset).
# 2. They're expected to vary within the same function call (e.g., the number of views,
#    which differs between context and target BatchedViews).


class OptimizationVariables(TypedDict, total=False):
    latent_set: Float[Tensor, "*batch latent_dim latent_num"]  # for Shape2VecSet, latent_dim=512, latent_num=8
    light_position: Float[Tensor, "*batch 3"]  
    object_pose: Float[Tensor, "*batch 6"] # [x, y, z, angle_x, angle_y, angle_z]
    object_scale: Float[Tensor, "*batch"] # scale factor of objects / depends on aabb

@dataclass
class OccVolume:
    grid: Float[Tensor, "*batch x y z coord"]
    occ_logits: Float[Tensor, "*batch x y z"] # logits \in (-inf, inf)



class RenderOutputs(TypedDict, total=False):
    shadow_map: Float[Tensor, "*batch H W"]
    queries_coords: Float[Tensor, "*batch H W D 3"] # rays are assumed to be [H, W]
    occ: Float[Tensor, "*batch 1 H W D"]