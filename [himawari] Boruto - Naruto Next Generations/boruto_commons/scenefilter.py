import numpy as np
import stgfunc as stg
import lvsfunc as lvf
import mvsfunc as mvf
import vsdpir as dpir
import EoEfunc as eoe
import vardefunc as vdf
import vapoursynth as vs
from vsutil import plane
from typing import Tuple
from .classes import Crop, SimpleCrop
from .masking import get_downdescale_mask
from .cropping import crop_resize_to_720p

core = vs.core

ENDING_15 = [
    SimpleCrop((0, 756)),
    SimpleCrop((767, 938)),
    SimpleCrop((948, 1137)),
    SimpleCrop((1278, 1662)),
    SimpleCrop((1850, 2144)),
    Crop((1663, 1777), 314, 315, 0, 0, False),
    Crop((1778, 1849), 257, 255),
    Crop((1138, 1184), 226, 244, 141, 638),
    Crop((1185, 1229), 211, 466, 135, 136),
    Crop((1230, 1277), 170, 146, 816, 87),
]


def filterEnding15(clip: vs.VideoNode, ED_FRAMES: Tuple[int, int]) -> vs.VideoNode:

  def _filterEnding(n: int, f: vs.VideoFrame) -> vs.VideoNode:
    ed_n = n - ED_FRAMES[0]
    if n not in np.arange(ED_FRAMES[0], ED_FRAMES[1] + 1):
      return clip[n]

    for i in np.arange(len(ENDING_15)):
      crop = ENDING_15[i]
      if ed_n in np.arange(crop.frames[0], crop.frames[1] + 1):
        return crop_resize_to_720p(clip, crop)[n]

    return clip[n]

  return core.std.FrameEval(clip, _filterEnding, clip)


def filterEnding16(clip: vs.VideoNode, ED_FRAMES: Tuple[int, int]) -> vs.VideoNode:
  def _filterEnding(n: int, f: vs.VideoFrame) -> vs.VideoNode:
    clip_y = plane(clip, 0)
    denoise_y = eoe.denoise.BM3D(clip_y, 1.45, 2, 'high')

    denoised = vdf.misc.merge_chroma(
        stg.denoise.KNLMeansCL(denoise_y, 1, 2, 4, 1.54, True, clip_y),
        stg.denoise.KNLMeansCL(clip, 1, 2, 8, [None, 1.14, 1.32], True, clip)
    )

    edfx = denoised.edgefixer.ContinuityFixer(*(4,) * 4)
    edfx = denoised.edgefixer.ContinuityFixer(*(3,) * 4)

    return edfx.resize.Spline64(1280, 720, format=vs.YUV444P16)

  return core.std.FrameEval(clip, _filterEnding, clip)


def filterTVTokyo(clip: vs.VideoNode, bil_downscale: vs.VideoNode, TV_TOKYO_FRAMES: Tuple[int, int]) -> vs.VideoNode:
  return lvf.rfs(
      clip,
      mvf.ToYUV(dpir.DPIR(mvf.ToRGB(bil_downscale, depth=32)), depth=16),
      TV_TOKYO_FRAMES
  )


def filterPreview(clip: vs.VideoNode, replace: vs.VideoNode, descale_mask: vs.VideoNode, ED_FRAMES: Tuple[int, int]) -> vs.VideoNode:
  downscaled_descale_mask = get_downdescale_mask(descale_mask)

  bil_masked = core.std.MaskedMerge(clip, replace, downscaled_descale_mask)

  return lvf.rfs(clip, bil_masked, (ED_FRAMES[1] + 1, clip.num_frames))