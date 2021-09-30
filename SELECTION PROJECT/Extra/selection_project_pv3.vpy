import vsdpir
import adjust
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
from vsutil import get_y, depth, get_depth, iterate
from vardefunc.misc import merge_chroma, DebugOutput
from vardefunc.util import finalise_output, initialise_input

core = vs.core

SRC_AVC = FileInfo(r"E:\Desktop\SELECTION PROJECT PV\PV03\SELECTION PROJECT_PV03_AVC.mp4", idx=stg.src)
SRC_VP9 = FileInfo(r"E:\Desktop\SELECTION PROJECT PV\PV03\SELECTION PROJECT_PV03_VP9.webm", idx=stg.src)


DEBUG = DebugOutput(num=9, props=7)

END_ARTWORK = (2644, None)
WEIRD_ASS_LINES = [(2125, 2179)]
LOW_AF_FILTERING = [(2180, 2238)]


class Filtering:
  @DEBUG.catch(op='@=')
  @finalise_output
  @initialise_input(bits=8, matrix=1, primaries=1, transfer=1)
  def main(self, src_avc: vs.VideoNode = SRC_AVC.clip_cut, src_vp9: vs.VideoNode = SRC_VP9.clip_cut) -> vs.VideoNode:
    src = self.splice_srcs(src_avc, src_vp9)

    denoise = self.denoise(src)

    denoise_fm = self.freqmerge(src, denoise)

    deblock = self.deblock_DPIR(denoise_fm)

    deblock = deblock.resize.Bicubic(format=vs.YUV420P16)

    contrasharp = self.contras(deblock, denoise_fm)

    dehalo = self.dehalo(contrasharp, deblock)

    dehalo = lvf.rfs(dehalo, depth(src, 16), LOW_AF_FILTERING)

    idk_lineart = self.wtffff(dehalo)

    pepega = self.pepegaendartworkfix(idk_lineart, deblock, denoise_fm)

    deband = self.deband(pepega)

    grain = self.grain(deband, 0.36, 8)

    return grain

  @finalise_output(bits=16)
  def splice_srcs(self, src_avc: vs.VideoNode, src_vp9: vs.VideoNode) -> vs.VideoNode:
    global DEBUG

    merge = lvf.rfs(src_vp9, src_avc, [
        (154, 192), (1007, 1105), (1857, 1894),
        (2330, 2341), (2551, 2643)
    ])

    freqmerge = self.freqmerge(src_vp9, src_avc)

    # No nose, no mouth LMAO
    merge = lvf.rfs(merge, freqmerge, LOW_AF_FILTERING)

    name_merge_1023 = src_avc.std.MaskedMerge(src_vp9, kgf.squaremask(src_avc, 770, 120, 961, 855))

    merge = lvf.rfs(merge, name_merge_1023, (1023, 1105))

    name_merge_1825 = src_avc.std.MaskedMerge(src_vp9, kgf.squaremask(src_avc, 225, 302, 1244, 257))

    merge = lvf.rfs(merge, name_merge_1825, (1825, 1856))

    merge = lvf.rfs(merge, src_avc.std.Merge(src_vp9, 0.45), (1958, 1983))

    DEBUG @= merge

    return merge

  def denoise(self, clip: vs.VideoNode) -> vs.VideoNode:
    y = get_y(clip)

    y_den = haf.SMDegrain(y, thSAD=240)

    y_sden = eoe.dn.BM3D(y_den, 1.8, 1, 'np')

    return stg.denoise.KNLMeansCL(merge_chroma(y_sden, clip), sigma=[0.12, 0.75])

  @finalise_output(bits=16)
  @initialise_input(bits=32)
  def deblock_DPIR(self, clip: vs.VideoNode) -> vs.VideoNode:
    denoise_rgb = mvf.ToRGB(clip)

    denoise_den = vsdpir.DPIR(denoise_rgb, 0.85, 'denoise')
    denoise_deb = vsdpir.DPIR(denoise_den, 20, 'deblock')

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

    return lvf.rfs(dehalo, ref, [(359, 410), END_ARTWORK])

  def deband(self, clip: vs.VideoNode) -> vs.VideoNode:
    clip_y = haf.FastLineDarkenMOD(get_y(clip), 120)

    linemask = stg.mask.linemask(clip_y)
    detail_mask = stg.mask.generate_detail_mask(clip_y)

    deb = dumb3kdb(clip, 8, 21)
    deb = dumb3kdb(deb, 16, 26)
    deb = dumb3kdb(deb, 20, 35)

    deb = lvf.rfs(deb, dumb3kdb(clip), [(272, 315)] + LOW_AF_FILTERING)

    return deb.std.MaskedMerge(clip, detail_mask).std.MaskedMerge(clip, linemask)

  def wtffff(self, clip: vs.VideoNode) -> vs.VideoNode:
    # Every encoder knows that ruining YouTube lineart in
    # some scenes actually makes it better, what is it
    # going to do? Be worse than YouTube's??

    clip_y = get_y(clip)

    lineart = clip_y.resize.Bicubic(clip.width / 2, clip.height / 2, filter_param_a=-1 / 2, filter_param_b=1 / 4)

    lineart = adjust.Tweak(lineart, 1, 0.5, -100)

    lineart = haf.FastLineDarkenMOD(lineart, 58, 2, 190, 2, 1)

    lineart = lineart.resize.Bicubic(clip.width, clip.height, filter_param_a=1 / 3, filter_param_b=1 / 3)

    mask = kgf.kirsch(clip_y)

    mask = stg.Maximum(mask, 3)

    mask = iterate(iterate(mask, core.std.Inflate, 5), core.std.Deflate, 2)

    mask = mask.std.BinarizeMask(16 << 8)

    mask = iterate(mask, lambda c: c.std.Maximum().std.Minimum(), 10)

    mask = iterate(iterate(mask, core.std.Minimum, 3), core.std.Deflate, 4)

    mask = mask.bilateral.Gaussian(1)

    wtflineart = clip_y.std.MaskedMerge(lineart, mask)

    return lvf.rfs(clip, merge_chroma(wtflineart, clip), WEIRD_ASS_LINES)

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

  def pepegaendartworkfix(self, clip: vs.VideoNode, deblock: vs.VideoNode, denoise: vs.VideoNode) -> vs.VideoNode:
    return lvf.rfs(clip, clip.std.MaskedMerge(
        deblock.std.Merge(denoise, 0.26), kgf.squaremask(clip, 697, 986, 128, 50)
    ), END_ARTWORK)

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
