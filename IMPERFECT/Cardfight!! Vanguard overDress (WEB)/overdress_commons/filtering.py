import vsdpir
import random
import EoEfunc as eoe
import lvsfunc as lvf
import vapoursynth as vs
from vsutil import get_y
from stgfunc import depth
from debandshit import dumb3kdb
from typing import Optional, Tuple
from vardautomation import FileInfo
from vardefunc.misc import merge_chroma
from lvsfunc.kernels import Catrom, Bicubic
from vardefunc.noise import Graigasm, AddGrain
from vardefunc.util import initialise_input, finalise_output

core = vs.core


class overDressFiltering():
  ending_only_credits: bool = False
  opening_ranges: Optional[Tuple[int]] = None
  ending_ranges: Optional[Tuple[int]] = None
  oshirase_ranges: Optional[Tuple[int]] = None

  def __init__(self, CR: FileInfo):
    self.CR = CR

  @finalise_output()
  def filterchain(self):
    src = depth(self.CR.clip_cut, 16)

    den = self.denoise(src)

    lines = self.unfuck_lines(den)

    unsharp = self.unsharpen(lines)

    if self.opening_ranges:
      unsharp = lvf.rfs(unsharp, lines, self.opening_ranges)

    deband = dumb3kdb(unsharp, 8, 24)

    grain = self.graining(deband)

    if self.oshirase_ranges:
      grain = lvf.rfs(grain, src, self.oshirase_ranges)

    if self.ending_ranges:
      replace_ed = src if self.ending_only_credits else den
      grain = lvf.rfs(grain, replace_ed, self.ending_ranges)

    return self.custom(src, den, lines, unsharp, deband, grain)

  def denoise(self, clip: vs.VideoNode) -> vs.VideoNode:
    return eoe.dn.BM3D(clip, 2.95, 1, 'fast')

  def unfuck_lines(self, clip: vs.VideoNode) -> vs.VideoNode:
    den_rgb = clip.resize.Bicubic(format=vs.RGBS, matrix_in=1)
    deb_rgb = vsdpir.DPIR(den_rgb, 30, 'deblock')

    return deb_rgb.resize.Bicubic(format=vs.YUV420P16, matrix=1)

  @finalise_output(bits=16)
  @initialise_input(bits=32)
  def unsharpen(self, clip: vs.VideoNode) -> vs.VideoNode:
    y = get_y(clip)

    upcy = Bicubic(-0.5, 0.25).scale(y, 1920 * 1.5, 1080 * 1.5)

    caty = Catrom().scale(upcy, 1920, 1080)

    return merge_chroma(caty, clip)

  def graining(self, clip: vs.VideoNode) -> vs.VideoNode:
    seed = random.seed()

    return Graigasm(
        thrs=[x << 8 for x in (26, 87, 200)],
        strengths=[(0.7, 0.2), (1.2, 0.085), (1.6, 0.05)],
        sizes=(1.23, 1.13, 1.05),
        sharps=(80, 40, 30),
        grainers=[
            AddGrain(seed=seed, constant=False),
            AddGrain(seed=seed, constant=True),
            AddGrain(seed=seed, constant=True)
        ]
    ).graining(clip)

  def custom(
      self, src: vs.VideoNode, den: vs.VideoNode, lines: vs.VideoNode,
      unsharp: vs.VideoNode, deband: vs.VideoNode, grain: vs.VideoNode
  ) -> vs.VideoNode:
    return grain
