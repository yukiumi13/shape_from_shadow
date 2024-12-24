from trimesh import PointCloud
from jaxtyping import Float
from torch import Tensor
from torchvision.utils import save_image
from typing import Union
from pathlib import Path

def export_pts(pts: Float[Tensor, "*batch 3"],
               save_path: Union[Path,str]):
    if isinstance(save_path, str):
        save_path = Path(save_path)
    if save_path.is_dir():
        save_path = save_path / 'xyz.ply'
    
    pts = pts.cpu().numpy().reshape(-1,3)
    pts3d = PointCloud(pts)
    pts3d.export(save_path)
    
def export_img(img: Float[Tensor, "*batch C H W"],
                save_path: Union[Path,str]):
    
    if isinstance(save_path, str):
        save_path = Path(save_path)
    if save_path.is_dir():
        save_path = save_path / 'vis.png'

    save_image(img.cpu(), save_path)
    