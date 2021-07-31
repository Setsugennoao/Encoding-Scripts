import stgfunc as stg
import lvsfunc as lvf
import EoEfunc as eoe
import havsfunc as haf
from typing import List
import vardefunc as vdf
import vapoursynth as vs
from vsutil import get_y, depth
from lvsfunc.types import Range, Union, Tuple
from .constants import descale_w, descale_h, kernel

core = vs.core


def rescale_aa(clip: vs.VideoNode, replace: Tuple[vs.VideoNode, Union[Range, List[Range], None]] = None, retTuple: bool = False) -> vs.VideoNode:  # pylint: disable=unsubscriptable-object
  clip_y = depth(get_y(clip), 32)

  descaled = kernel.descale(clip_y, descale_w, descale_h)
  rescaled = kernel.scale(descaled, clip.width, clip.height)

  descale_mask = lvf.scale.descale_detail_mask(clip_y, rescaled)

  upscaled = stg.upscale.upscale(descaled, clip.width, clip.height)

  upscaled = core.std.MaskedMerge(upscaled, clip_y, descale_mask)
  upscaled = depth(upscaled, 16)

  rescaled = vdf.misc.merge_chroma(upscaled, clip)

  if replace is not None:
    rescaled = lvf.rfs(rescaled, replace[0], replace[1])

  aa = lvf.sraa(rescaled, rfactor=2)

  return (aa, rescaled) if retTuple else aa


def temp_degrain(clip: vs.VideoNode, search: int = 2, **kwargs) -> vs.VideoNode:
  return eoe.denoise.CMDegrain(clip, 2, 150, prefilter=3, search=search, contrasharp=True, RefineMotion=True, **kwargs)


def debanding(clip: vs.VideoNode, ref: vs.VideoNode = None) -> Tuple[vs.VideoNode, vs.VideoNode]:
  deband = core.average.Mean([
      vdf.deband.dumb3kdb(clip, 8, 25),
      vdf.deband.dumb3kdb(clip, 12, 36),
      vdf.deband.dumb3kdb(clip, 24, 48),
      vdf.placebo.deband(clip, 8, 4, 1, 0),
  ])

  if ref is not None:
    deband = eoe.misc.ContraSharpening(deband, ref)

  deband = core.f3kdb.Deband(deband, 16, 0, 0, 0, 32, 16)

  return (deband, get_y(deband))


def denoising(clip: vs.VideoNode, ref: vs.VideoNode = None) -> Tuple[vs.VideoNode, vs.VideoNode]:
  denoise = eoe.denoise.BM3D(clip, 2, ref=ref)
  denoise = haf.FastLineDarkenMOD(denoise, strength=8, protection=4, luma_cap=255 - (17500 >> 8), threshold=2, thinning=0)
  return (denoise, get_y(denoise))


def continuityFixer(clip: vs.VideoNode) -> vs.VideoNode:
  return clip.edgefixer.ContinuityFixer(0, 3, 0, 3)
