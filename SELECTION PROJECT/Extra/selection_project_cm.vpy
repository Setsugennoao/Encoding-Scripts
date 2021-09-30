import vsdpir
import numpy as np
import G41Fun as g4f
import stgfunc as stg
import EoEfunc as eoe
import mvsfunc as mvf
import lvsfunc as lvf
import havsfunc as haf
import kagefunc as kgf
import vapoursynth as vs
from debandshit import dumb3kdb
from vardautomation import FileInfo
from vsutil import get_y, depth, get_depth
from vardefunc.misc import merge_chroma, DebugOutput
from vardefunc.util import finalise_output, initialise_input

core = vs.core

SRC_AVC = FileInfo(r"E:\Desktop\SELECTION PROJECT PV\CM\SELECTION PROJECT_CM_AVC.mp4", idx=stg.src)
SRC_VP9 = FileInfo(r"E:\Desktop\SELECTION PROJECT PV\CM\SELECTION PROJECT_CM_VP9.webm", idx=stg.src)


DEBUG = DebugOutput(num=9, props=7)


class Filtering:
  @DEBUG.catch(op='@=')
  @finalise_output
  @initialise_input(bits=8, matrix=1, primaries=1, transfer=1)
  def main(self, src_avc: vs.VideoNode = SRC_AVC.clip_cut, src_vp9: vs.VideoNode = SRC_VP9.clip_cut) -> vs.VideoNode:
    src = self.splice_srcs(src_avc, src_vp9)

    denoise = self.denoise(src)

    denoise_frqm = self.freqmerge(denoise, src)

    deblock = self.deblock_DPIR(denoise_frqm)

    deblock = deblock.resize.Bicubic(format=vs.YUV420P16)

    contrasharp = self.contras(deblock, denoise_frqm)

    contrasharp = lvf.rfs(contrasharp, deblock, (300, None))

    dehalo = self.dehalo(contrasharp, deblock)

    deband = self.deband(dehalo)

    deband_frqm = self.freqmerge(deband, src)

    bandMask = kgf.adaptive_grain(deband_frqm, luma_scaling=6.5, show_mask=True)

    deband_frqm = deband_frqm.std.MaskedMerge(deband, bandMask)

    deband_frqm = lvf.rfs(deband_frqm, deband, (None, 30))

    grain = self.grain(deband_frqm, 0.36, 8)

    return grain

  @finalise_output(bits=16)
  def splice_srcs(self, src_avc: vs.VideoNode, src_vp9: vs.VideoNode) -> vs.VideoNode:
    global DEBUG

    merge = src_avc

    merge = lvf.rfs(src_vp9, src_avc, (171, 195))

    # DEBUG @= src_avc
    # DEBUG @= src_vp9
    DEBUG @= merge

    return merge

  def denoise(self, clip: vs.VideoNode) -> vs.VideoNode:
    y = get_y(clip)

    y_den = haf.SMDegrain(y, thSAD=240)

    y_sden = eoe.dn.BM3D(y_den, 1.44, 1, 'np')

    dffsigma = 14

    sloc = [0, 0, 0.25, 0, 0.45, dffsigma / 8, 0.65, dffsigma / 4, 0.7, dffsigma / 1.5, 1, dffsigma]

    y_dfden = core.dfttest.DFTTest(y_sden, sigma=dffsigma / 4, ssx=sloc, ssy=sloc)

    return stg.denoise.KNLMeansCL(merge_chroma(y_dfden, clip), sigma=[0.23, 0.75])

  @finalise_output(bits=16)
  @initialise_input(bits=32)
  def deblock_DPIR(self, clip: vs.VideoNode) -> vs.VideoNode:
    denoise_rgb = mvf.ToRGB(clip)

    denoise_deb = vsdpir.DPIR(denoise_rgb, 42.75, 'deblock')

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
    return g4f.MaskedDHA(clip, 2, 2, 0.04, 0.96, 50, 200)

  def deband(self, clip: vs.VideoNode) -> vs.VideoNode:
    clip_y = haf.FastLineDarkenMOD(get_y(clip), 120)

    linemask = stg.mask.linemask(clip_y)
    detail_mask = stg.mask.generate_detail_mask(clip_y)

    deb = dumb3kdb(clip)
    deb = dumb3kdb(deb, 8, 21)
    deb = dumb3kdb(deb, 16, 26)
    deb = dumb3kdb(deb, 20, 35)

    return deb.std.MaskedMerge(clip, detail_mask).std.MaskedMerge(clip, linemask)

  def freqmerge(self, clip: vs.VideoNode, filtered: vs.VideoNode) -> vs.VideoNode:
    Oycore = stg.oyster.Core()

    clip_y, filt_y = depth(get_y(clip), 32), depth(get_y(filtered), 32)

    ref = stg.oyster.Basic(clip_y, None)

    mask = Oycore.GenBlockMask(clip_y)

    lowpass = [0.0, 0.0, 0.12, 1024.0, 1.0, 1024.0]

    ref = Oycore.FreqMerge(filt_y, ref, 9, lowpass)
    mer_y = Oycore.FreqMerge(filt_y, clip_y, 9, lowpass)

    freqmerged = core.std.MaskedMerge(mer_y, ref, mask)

    return merge_chroma(depth(freqmerged, get_depth(clip)), clip)

  def grain(self, clip: vs.VideoNode, amount: float, luma_scaling: float) -> vs.VideoNode:
    grain = clip.f3kdb.Deband(8, 0, 0, 0, amount * 50 if amount < 1 else amount * 5, 0)

    grain_mask = core.adg.Mask(clip.std.PlaneStats(), luma_scaling)

    grained = core.grain.Add(grain, var=amount, constant=True)

    return core.std.MaskedMerge(grain, grained, grain_mask)


if __name__ == '__vapoursynth__':
  grain = Filtering().main()
  grain.set_output(0)
else:
  Filtering().main()
