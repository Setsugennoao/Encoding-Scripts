import kagefunc as kgf
import havsfunc as haf
import vapoursynth as vs
from vsutil import get_y, depth
from vardautomation import FileInfo
from vardefunc.misc import merge_chroma
from vardefunc.util import finalise_output
from vsdenoise.bm3d import BM3D, BM3DCuda, Profile

core = vs.core

core.max_cache_size = 16 * 2 ** 10


class KyouNoAquaaFiltering:
  YOUTUBE: FileInfo

  def __init__(self, YOUTUBE: FileInfo):
    self.YOUTUBE = YOUTUBE

  @finalise_output()
  def filterchain(self):
    src = self.mix_source()

    y = depth(get_y(src), 16)
    src_444 = src.resize.Bicubic(format=vs.YUV444P16)

    deblock_mask = kgf.kirsch(self.get_dctmask(y, 0.5))
    denoise_mask = self.get_dctmask(y, 0.34).std.BinarizeMask(27 << 8).std.Deflate().std.Inflate().std.Inflate().std.Deflate()

    deblock = self.deblock(src_444)
    denoise = self.denoise(y)

    masked = src_444.std.MaskedMerge(deblock, deblock_mask)

    masked = masked.std.MaskedMerge(merge_chroma(denoise, src_444), denoise_mask)

    masked = masked.resize.Bicubic(
        format=vs.YUV420P16, filter_param_a_uv=-0.5, filter_param_b_uv=0.25
    )

    grain = self.graining(masked, denoise_mask)

    return grain

  def mix_source(self):
    return self.YOUTUBE.clip_cut

  def get_dctmask(self, y: vs.VideoNode, lumi: float):
    return y.dctf.DCTFilter((lumi, 1, 0.75, 0.25, 0.5, 0.5, 0.12, 0))

  def deblock(self, clip: vs.VideoNode) -> vs.VideoNode:
    return haf.SMDegrain(clip, 2, 750, 250, True, True)

  def denoise(self, y: vs.VideoNode) -> vs.VideoNode:
    try:
      return BM3DCuda(y, 2.44, 1, Profile.LC, refine=2).clip
    except vs.Error:
      return BM3D(y, 1.356, 1, Profile.LC, refine=2).clip

  def graining(self, clip: vs.VideoNode, mask: vs.VideoNode) -> vs.VideoNode:
    return clip.grain.Add(0.37, 0.01).std.MaskedMerge(clip, mask)
