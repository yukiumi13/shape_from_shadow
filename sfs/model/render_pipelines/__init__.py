# Copyright (c) 2024 Yang Li, Microsoft. All rights reserved.
# MIT License.
# 
# --------------------------------------------------------
# Implementation of Shadow Rendering
# --------------------------------------------------------
# 
# Created on Sun Dec 22 2024.

from typing import Optional, Union
from .base_render import RenderPipeline
from .shadow_render import ShadowRender, ShadowRenderCfg

RENDER_PIPELINES = {
    "simple_shadow_render": ShadowRender,
}

RenderPipelineCfg = Union[ShadowRenderCfg]
ImplementedRenderPipelines = Union[ShadowRender] # For Linting

def get_renderer(cfg: RenderPipelineCfg) -> ImplementedRenderPipelines:
    render_pipe = RENDER_PIPELINES[cfg.name]
    render_pipe = render_pipe(cfg)
    return render_pipe