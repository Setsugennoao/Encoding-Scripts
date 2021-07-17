import stgfunc as stg
import kagefunc as kgf
import vapoursynth as vs
from typing import Tuple

core = vs.core


def get_detail_mask(clip_y: vs.VideoNode) -> vs.VideoNode:
  return core.std.Expr([
      stg.mask.generate_detail_mask(clip_y, 0.013),
      kgf.kirsch(clip_y)
  ], "x y -")


def get_linemasks(clip_y: vs.VideoNode) -> Tuple[vs.VideoNode, vs.VideoNode]:
  lineart_hard = stg.mask.linemask(clip_y)
  lineart_hard = lineart_hard.std.Binarize(75 << 8).std.Minimum().std.Deflate()

  lineart_light = kgf.kirsch(clip_y).std.Binarize(255 << 8)

  return (lineart_hard, lineart_light)
