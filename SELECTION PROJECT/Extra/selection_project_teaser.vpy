import vsdpir
import G41Fun as g4f
import stgfunc as stg
import EoEfunc as eoe
import mvsfunc as mvf
import lvsfunc as lvf
import havsfunc as haf
import vapoursynth as vs
from vsutil import get_y
from functools import partial
from debandshit import dumb3kdb
from vardautomation import FileInfo
from vardefunc.misc import merge_chroma, DebugOutput
from vardefunc.util import finalise_output, initialise_input

core = vs.core

SRC_AVC = FileInfo(r"E:\Desktop\SELECTION PROJECT PV\Teaser\SELECTION PROJECT_TEASER_AVC.mkv", idx=partial(stg.src, depth=16))
SRC_VP9 = FileInfo(r"E:\Desktop\SELECTION PROJECT PV\Teaser\SELECTION PROJECT_TEASER_VP9.mkv", idx=partial(stg.src, depth=16))

DEBUG = DebugOutput(SRC_AVC.clip_cut, num=9, props=7)


class Filtering:
  @DEBUG.catch(op='@=')
  @finalise_output
  @initialise_input(matrix=1, primaries=1, transfer=1)
  def main(self, src_avc: vs.VideoNode = SRC_AVC.clip_cut, src_vp9: vs.VideoNode = SRC_VP9.clip_cut) -> vs.VideoNode:
    src = self.splice_srcs(src_avc, src_vp9)
    src = src.resize.Bicubic(format=vs.YUV444P16)

    y = get_y(src)

    y_den = self.SMDegrain_Y(y)

    y_sden = self.BM3D_Y(y_den)

    merge = merge_chroma(y_sden, src)

    denoise = self.denoise_KNL(merge)

    deblock = self.deblock_DPIR(denoise)

    dehalo = self.dehalo(deblock)

    deband = dumb3kdb(dehalo, 8, 24)

    return self.grain(deband, 0.36, 8).resize.Bicubic(format=vs.YUV420P16)

  def splice_srcs(self, src_avc: vs.VideoNode, src_vp9: vs.VideoNode) -> vs.VideoNode:
    return lvf.rfs(src_avc, src_vp9, (481, 516))

  @initialise_input(bits=16)
  def SMDegrain_Y(self, y: vs.VideoNode) -> vs.VideoNode:
    y_den = haf.SMDegrain(y, thSAD=300, contrasharp=True)
    y_den = haf.SMDegrain(y_den, thSAD=260, contrasharp=True)
    y_den = haf.SMDegrain(y_den, thSAD=140, contrasharp=True)

    return y_den

  def BM3D_Y(self, y: vs.VideoNode) -> vs.VideoNode:
    return eoe.dn.BM3D(y, 2, 1, 'high')

  def denoise_KNL(self, clip: vs.VideoNode) -> vs.VideoNode:
    return stg.denoise.KNLMeansCL(clip, sigma=[0.15, 0.75])

  @finalise_output(bits=16)
  @initialise_input(bits=32)
  def deblock_DPIR(self, clip: vs.VideoNode) -> vs.VideoNode:
    denoise_rgb = mvf.ToRGB(clip)
    denoise_deb = vsdpir.DPIR(denoise_rgb, 30, 'deblock')
    denoise_den = eoe.dn.BM3D(denoise_deb, 0.5, 1, 'lc')
    return mvf.ToYUV(denoise_den)

  def dehalo(self, clip: vs.VideoNode) -> vs.VideoNode:
    return g4f.MaskedDHA(clip, 2, 2, 0.04, 0.96, 50, 200)

  def grain(self, clip: vs.VideoNode, amount: float, luma_scaling: float) -> vs.VideoNode:
    grain = clip.f3kdb.Deband(8, 0, 0, 0, amount * 50 if amount < 1 else amount * 5, 0)

    grain_mask = core.adg.Mask(clip.std.PlaneStats(), luma_scaling)

    grained = core.grain.Add(grain, var=amount, constant=True)

    return core.std.MaskedMerge(grain, grained, grain_mask)


if __name__ == '__vapoursynth__':
  del DEBUG
  grain = Filtering().main()
  grain.set_output()
else:
  Filtering().main()
