import vapoursynth as vs
from vsutil import iterate, depth

core = vs.core


def get_downdescale_mask(descale_mask: vs.VideoNode) -> vs.VideoNode:
  return iterate(iterate(depth(descale_mask.resize.Bicubic(1280, 720), 16), core.std.Inflate, 20).std.Binarize(24 << 8), core.std.Inflate, 15).std.Binarize(24 << 8)
