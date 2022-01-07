import G41Fun as g4f
import EoEfunc as eoe
import havsfunc as hvf
import vapoursynth as vs
from vsutil import join, split
from .constants import degrain_args, knl_args


def dehalo_filter(clip: vs.VideoNode) -> vs.VideoNode:
  return g4f.MaskedDHA(clip, 2, 2, 0.50, 1.0)


def denoise_filter(clip: vs.VideoNode, ref: vs.VideoNode) -> vs.VideoNode:
  denoise = eoe.denoise.BM3D(clip, [2.33, 1.35, 1.3], 1, ["vn", "high"])

  return eoe.misc.ContraSharpening(denoise, ref)


def degrain_filter(clip: vs.VideoNode) -> vs.VideoNode:
  planes = enumerate(split(clip))

  def _degrain_plane(plane_clip: vs.VideoNode, plane_index: int):
    return hvf.SMDegrain(plane_clip, 2, *degrain_args[plane_index])

  return join([_degrain_plane(x, i) for i, x in planes])


def knl_filter(clip: vs.VideoNode) -> vs.VideoNode:
  planes = split(clip)

  def _knl_plane(plane_clip: vs.VideoNode, plane_index: int) -> vs.VideoNode:
    return plane_clip.knlm.KNLMeansCL(**knl_args[plane_index])

  joined = [_knl_plane(x, i) for i, x in enumerate(planes)]

  return join(joined)
