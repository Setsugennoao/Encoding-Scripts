import vsdpir
import numpy as np
import G41Fun as g4f
import stgfunc as stg
import EoEfunc as eoe
import mvsfunc as mvf
import lvsfunc as lvf
import kagefunc as kgf
import havsfunc as haf
import vapoursynth as vs
from debandshit import dumb3kdb
from vardautomation import FileInfo
from vsutil import get_y, insert_clip
from vardefunc.misc import merge_chroma, DebugOutput
from vardefunc.util import finalise_output, initialise_input

core = vs.core

SRC_AVC = FileInfo(r"E:\Desktop\SELECTION PROJECT PV\PV02\SELECTION PROJECT_PV02_AVC.mkv", idx=stg.src)
SRC_VP9 = FileInfo(r"E:\Desktop\SELECTION PROJECT PV\PV02\SELECTION PROJECT_PV02_VP9.mkv", idx=stg.src)
SRC_AV1 = FileInfo(r"E:\Desktop\SELECTION PROJECT PV\PV02\SELECTION PROJECT_PV02_AV1.mkv", idx=stg.src)

DEBUG = DebugOutput(SRC_AVC.clip_cut, num=9, props=7)


class Filtering:
  @DEBUG.catch(op='@=')
  @finalise_output
  @initialise_input(bits=8, matrix=1, primaries=1, transfer=1)
  def main(
      self, src_avc: vs.VideoNode = SRC_AVC.clip_cut,
      src_vp9: vs.VideoNode = SRC_VP9.clip_cut,
      src_av1: vs.VideoNode = SRC_AV1.clip_cut
  ) -> vs.VideoNode:
    src = self.splice_srcs(src_avc, src_vp9, src_av1)

    denoise = self.denoise(src)

    deblock = self.deblock_DPIR(denoise)

    contrasharp = self.contras(deblock, denoise)

    dehalo = self.dehalo(contrasharp, deblock)

    fixartwork = self.pepegaendartworkfix(dehalo, denoise)

    deband = dumb3kdb(fixartwork, 8, 26)

    grain = self.grain(deband, 0.36, 8)

    return grain.resize.Bicubic(format=vs.YUV420P16)

  @finalise_output(bits=16)
  def splice_srcs(self, src_avc: vs.VideoNode, src_vp9: vs.VideoNode, src_av1: vs.VideoNode) -> vs.VideoNode:
    PAN = (729, 739)

    pan = core.std.FrameEval(
        core.std.BlankClip(src_av1, length=10),
        lambda n, f: core.std.Merge(src_av1[slice(*PAN)], src_vp9[slice(*PAN)], weight=n / 10),
        src_av1
    )

    merge = lvf.rfs(src_avc, src_vp9, [
        (606, 644), (PAN[1], 758), (1006, 1153),
        (1198, 1329), (1562, 1591), (1673, 1725),
    ])

    merge = lvf.rfs(merge, src_av1, [
        (None, 497), (539, 605), (645, PAN[0] + 1),
        (759, 787), (842, 1005), (1154, 1197),
        (1330, 1561), (1592, 1672), (1726, None)
    ])

    return insert_clip(merge, pan, PAN[0])

  def denoise(self, clip: vs.VideoNode) -> vs.VideoNode:
    y = get_y(clip)

    y_den = haf.SMDegrain(y, thSAD=24, contrasharp=True)

    y_sden = eoe.dn.BM3D(y_den, 2, 1, 'high')

    return stg.denoise.KNLMeansCL(merge_chroma(y_sden, clip), sigma=[None, 0.75])

  @finalise_output(bits=16)
  @initialise_input(bits=32)
  def deblock_DPIR(self, clip: vs.VideoNode) -> vs.VideoNode:
    denoise_rgb = mvf.ToRGB(clip)

    denoise_den = vsdpir.DPIR(denoise_rgb, 0.95, 'denoise')
    denoise_deb = vsdpir.DPIR(denoise_den, 30, 'deblock')

    return mvf.ToYUV(denoise_deb)

  def contras(self, clip: vs.VideoNode, ref: vs.VideoNode) -> vs.VideoNode:
    deb_y = get_y(clip)

    all_matrices = [[x, x / 2, x / 4] for x in np.arange(1, 2 ** 10)]
    error = [sum([abs(x - round(x)) for x in matrix[1:]]) for matrix in all_matrices]
    matrix = [round(all_matrices[error.index(min(error))][x]) for x in (2, 1, 2, 1, 0, 1, 2, 1, 2)]

    contra_y = core.std.Expr([
        eoe.misc.ContraSharpening(deb_y, get_y(ref)),
        deb_y.std.MergeDiff(deb_y.std.MakeDiff(deb_y.std.Convolution(matrix))),
    ], 'x dup + x y min max x y max min')

    contra_y = haf.FastLineDarkenMOD(contra_y, 22, 4, 185, 5, 1)

    return merge_chroma(contra_y, clip)

  def dehalo(self, clip: vs.VideoNode, ref: vs.VideoNode) -> vs.VideoNode:
    dehalo = g4f.MaskedDHA(clip, 2, 2, 0.04, 0.96, 50, 200)

    return lvf.rfs(dehalo, ref, [(306, 419), (2131, None)])

  def pepegaendartworkfix(self, clip: vs.VideoNode, denoise: vs.VideoNode) -> vs.VideoNode:
    den_444 = denoise.resize.Bicubic(format=clip.format.id)

    fixartwork = lvf.rfs(clip, clip.std.MaskedMerge(
        clip.std.Merge(den_444, 0.26), kgf.squaremask(clip, 697, 986, 128, 50)
    ), (2131, 2198))

    fixartwork = lvf.rfs(fixartwork, clip.std.MaskedMerge(
        clip.std.Merge(den_444, 0.34), kgf.squaremask(clip, 728, 994, 102, 43)
    ), (2287, 2345))

    return fixartwork.std.FreezeFrames(2346, fixartwork.num_frames - 1, fixartwork.num_frames - 1)

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
