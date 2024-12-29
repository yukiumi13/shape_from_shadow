# Shape-from-shadow
This is the implementation of a neural shape-from-shadow model, i.e., learning the shape given a shadow map.
## Modules
Generally, the repo comprises two modules with some modifications:
* 3D VAE
    * Occupancy Network -> Shape2VecSet 
* Shadow renderer
    * max_pool(OccNet(x)) -> accelerated occ volume rendering

The detailed introduction and demo of each module can be found at [demo.py](https://github.com/yukiumi13/sfs/blob/main/demo.ipynb).
## Acknowledges
The repo is built heavily upon [nerf_template](https://github.com/ashawkey/nerf_template) and [3DShape2VecSet](https://github.com/1zb/3DShape2VecSet). Thank these fancy projects.
