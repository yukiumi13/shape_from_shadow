import torch
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras, PointLights, RasterizationSettings, MeshRenderer, MeshRasterizer, HardPhongShader, look_at_view_transform
)
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
mesh = load_objs_as_meshes(["teapot.obj"], device=device)


bbox_min = mesh.verts_packed().min(0)[0]
bbox_max = mesh.verts_packed().max(0)[0]
bbox_center = (bbox_min + bbox_max) / 2
bbox_size = (bbox_max - bbox_min).max().item()


R, T = look_at_view_transform(dist=bbox_size * 2, elev=20, azim=30, at=bbox_center.unsqueeze(0))
cameras = FoVPerspectiveCameras(device=device, R=R, T=T)


lights = PointLights(device=device, location=[[0.0, 0.0, bbox_size * 2]])

raster_settings = RasterizationSettings(
    image_size=512, 
    blur_radius=0.0, 
    faces_per_pixel=1, 
)


renderer = MeshRenderer(
    rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
    shader=HardPhongShader(device=device, cameras=cameras, lights=lights)
)


images = renderer(mesh)
image = images[0, ..., :3].cpu().numpy()


gray_image = image.mean(axis=-1)
shadow_image = 1 - gray_image

plt.imshow(shadow_image, cmap='gray')
plt.title("Shadow Mask")
plt.axis('off')
plt.savefig('shadow.png')








