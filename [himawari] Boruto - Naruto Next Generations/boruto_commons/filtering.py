import stgfunc as stg
import lvsfunc as lvf
import EoEfunc as eoe
import vardefunc as vdf
import vapoursynth as vs
from vsutil import get_y, depth
from typing import Literal, Tuple
from .constants import desc_w, desc_h, kernel

core = vs.core


def bil_downscale(clip: vs.VideoNode) -> vs.VideoNode:
  return clip.resize.Bilinear(1280, 720, format=vs.YUV444P16)


def descale_denoiseY_filter(clip: vs.VideoNode) -> Tuple[vs.VideoNode, vs.VideoNode]:
  src_y = get_y(clip)

  clip_y = eoe.denoise.BM3D(src_y, 3.55, 1, "high", CUDA=[False, True])
  clip_y = eoe.misc.ContraSharpening(clip_y, src_y)
  clip_y = depth(clip_y, 32)

  descaled = kernel.descale(clip_y, desc_w, desc_h)
  krescaled = kernel.scale(descaled, 1920, 1080)

  descale_mask = lvf.scale.descale_detail_mask(clip_y, krescaled)

  upscaled = stg.upscale.upscale(descaled, desc_w * 2, desc_h * 2)

  rescaled = upscaled.resize.Spline64(1920, 1080)

  rescaled = core.std.MaskedMerge(rescaled, clip_y, descale_mask)

  return rescaled, descale_mask


def knl_filter(clip: vs.VideoNode, planes: Literal['Y', 'UV']) -> vs.VideoNode:
  sigma = 1.15 if planes == 'Y' else [None, 1.215]

  return stg.denoise.KNLMeansCL(clip, 2, 2, 4, sigma, True, clip)


def debanding_filter(clip: vs.VideoNode) -> vs.VideoNode:
  return core.f3kdb.Deband(
      core.average.Mean([
          vdf.deband.dumb3kdb(clip, 8, 25),
          vdf.placebo.deband(clip, 16, 3, 1, 0),
          vdf.placebo.deband(clip, 24, 4, 1, 0),
      ]), 16, 0, 0, 0, 16, 16
  )


def limit_yuv_filter(clip: vs.VideoNode) -> vs.VideoNode:
  return clip.std.Limiter(16 << 2, [235 << 2, 240 << 2], [0, 1, 2])
