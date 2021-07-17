import EoEfunc as eoe
import lvsfunc as lvf
import havsfunc as hvf
import vardefunc as vdf
import vapoursynth as vs
from typing import Tuple
from vsutil import join, split, plane
from .constants import degrain_args, light_knl_args, heavy_knl_args

core = vs.core


def sraa_filter(clip: vs.VideoNode) -> Tuple[vs.VideoNode, vs.VideoNode]:
  sraa = lvf.sraa(clip, 2)
  return (sraa, plane(sraa, 0))


def denoise_filter(clip: vs.VideoNode, ref: vs.VideoNode) -> Tuple[vs.VideoNode, vs.VideoNode]:
  denoise = eoe.denoise.BM3D(clip, [2.75, 1.75, 1.75], 1, "np", ref=ref)
  return (denoise, plane(denoise, 0))


def degrain_filter(clip: vs.VideoNode) -> vs.VideoNode:
  planes = enumerate(split(clip))

  def _degrain_plane(plane_clip: vs.VideoNode, plane_index: int):
    return hvf.SMDegrain(plane_clip, 2, *degrain_args[plane_index])

  return join([_degrain_plane(x, i) for i, x in planes])


def knl_filter(clip: vs.VideoNode, degrain: vs.VideoNode) -> Tuple[vs.VideoNode, vs.VideoNode]:
  planes = split(clip)
  degrain_planes = split(degrain)

  def _knl_filter(args) -> vs.VideoNode:
    def _knl_plane(plane_clip: vs.VideoNode, plane_degrain: vs.VideoNode, plane_index: int) -> vs.VideoNode:
      knl = plane_degrain.knlm.KNLMeansCL(**args[plane_index])
      return eoe.misc.ContraSharpening(knl, plane_clip, 2)

    joined = [_knl_plane(x, degrain_planes[i], i) for i, x in enumerate(planes)]

    return join(joined)

  return (_knl_filter(light_knl_args), _knl_filter(heavy_knl_args))


def debanding_filter(clip: vs.VideoNode) -> vs.VideoNode:
  return core.f3kdb.Deband(
      core.average.Mean([
          vdf.deband.dumb3kdb(clip, 8, 25),
          vdf.deband.dumb3kdb(clip, 12, 36),
          vdf.placebo.deband(clip, 16, 3, 1, 0),
          vdf.placebo.deband(clip, 32, 7, 1, 0),
      ]), 16, 0, 0, 0, 16, 16
  )
