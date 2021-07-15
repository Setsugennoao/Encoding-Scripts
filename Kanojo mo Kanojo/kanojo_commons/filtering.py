import stgfunc as stg
import lvsfunc as lvf
import EoEfunc as eoe
import havsfunc as haf
from typing import List
import vardefunc as vdf
import vapoursynth as vs
from vsutil import depth
from vsutil.clips import get_y
from lvsfunc.types import Range, Union, Tuple
from .constants import descale_w, descale_h, kernel

core = vs.core


def rescale_aa(clip: vs.VideoNode, replace: Tuple[vs.VideoNode, Union[Range, List[Range], None]] = None, retTuple: bool = False) -> vs.VideoNode:  # pylint: disable=unsubscriptable-object
  rescaled = lvf.scale.descale(clip, lambda c, w, y: stg.upscale.upscale_rescale(clip, c, w, y), descale_w, descale_h, kernel)  # pylint: disable=too-many-function-args
  rescaled = depth(rescaled, 16)

  if replace is not None:
    rescaled = lvf.rfs(rescaled, replace[0], replace[1])

  aa = lvf.sraa(rescaled, rfactor=2)

  return (aa, rescaled) if retTuple else aa


def temp_degrain(clip: vs.VideoNode, search: int = 2, **kwargs) -> vs.VideoNode:
  return eoe.denoise.CMDegrain(clip, 2, 150, prefilter=3, search=search, contrasharp=True, RefineMotion=True, **kwargs)


def debanding(clip: vs.VideoNode, ref: vs.VideoNode = None) -> Tuple[vs.VideoNode, vs.VideoNode]:
  deband = core.average.Mean([
      clip,
      vdf.deband.dumb3kdb(clip, 8, 25),
      vdf.deband.dumb3kdb(clip, 12, 36),
      vdf.deband.dumb3kdb(clip, 18, 64),
      vdf.deband.dumb3kdb(clip, 24, 96),
      vdf.placebo.deband(clip, 8, 4, 1, 0),
      vdf.placebo.deband(clip, 16, 3, 1, 0),
      vdf.placebo.deband(clip, 32, 7, 1, 0),
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
