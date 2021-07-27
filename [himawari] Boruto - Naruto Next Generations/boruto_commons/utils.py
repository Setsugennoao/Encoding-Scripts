import vapoursynth as vs
from vsutil import depth


def resize_spline(clip: vs.VideoNode) -> vs.VideoNode:
  return depth(clip.resize.Spline64(1280, 720), 16)
