import vsdpir
import stgfunc as stg
import EoEfunc as eoe
import mvsfunc as mvf
import lvsfunc as lvf
import kagefunc as kgf
import havsfunc as haf
import vapoursynth as vs
from typing import Tuple
from debandshit import dumb3kdb
from vardautomation import FileInfo
from vsutil import get_y, depth, get_depth
from vardefunc.misc import merge_chroma, DebugOutput
from vardefunc.util import finalise_output, initialise_input

core = vs.core

SRC_AVC = FileInfo(r"E:\Desktop\SELECTION PROJECT PV\PV01\SELECTION PROJECT_PV01_AVC.mkv", idx=stg.src)

DEBUG = DebugOutput(SRC_AVC.clip_cut, num=9, props=7)


class Filtering:
  @DEBUG.catch(op='@=')
  @finalise_output
  @initialise_input(bits=32, matrix=1, primaries=1, transfer=1)
  def main(self, src: vs.VideoNode = SRC_AVC.clip_cut) -> vs.VideoNode:
    src = src.resize.Spline64(format=vs.YUV444PS)

    denoise, sdenoise = self.denoise_SBM3NL(src)

    deblock, sdeblock = self.denoise_DPIR(denoise, sdenoise)

    deblock = self.merge_lineart(deblock, sdeblock)

    sfilt = self.sfilt(deblock, sdeblock)

    rep = self.rep_csharp(sfilt)

    pepega = self.pepegascenefilteringffs(rep, deblock, denoise)

    pepega = depth(pepega, 16)

    deband = dumb3kdb(pepega, 8, 25)

    grain = self.grain(deband)

    return grain.resize.Bicubic(format=vs.YUV420P16)

  def denoise_SBM3NL(self, clip: vs.VideoNode) -> Tuple[vs.VideoNode, vs.VideoNode]:
    denoise = eoe.dn.BM3D(clip, 1.7, 1, 'np', chroma=True, CUDA=False)

    y = get_y(clip)

    y_den = depth(haf.SMDegrain(depth(y, 16), thSAD=50, prefilter=3, RefineMotion=True, contrasharp=True), get_depth(y))

    y_sden = eoe.dn.BM3D(y_den, 3.56, 1, 'np', CUDA=False)

    sdenoise = merge_chroma(y_sden, denoise)

    sdenoise = stg.denoise.KNLMeansCL(sdenoise, sigma=0.75)

    return denoise, sdenoise

  @initialise_input(bits=32)
  def denoise_DPIR(self, light: vs.VideoNode, strong: vs.VideoNode) -> Tuple[vs.VideoNode, vs.VideoNode]:
    light_rgb, strong_rgb = mvf.ToRGB(light), mvf.ToRGB(strong)

    deblock_deb = vsdpir.DPIR(light_rgb, 35, 'deblock')

    strong_den = vsdpir.DPIR(strong_rgb, 10, 'denoise')
    strong_deb = vsdpir.DPIR(strong_den, 55, 'deblock')

    return mvf.ToYUV(deblock_deb), mvf.ToYUV(strong_deb)

  def merge_lineart(self, light: vs.VideoNode, strong: vs.VideoNode) -> vs.VideoNode:
    y = depth(get_y(light), 16)

    linemask = stg.mask.tcanny(y, 0.0155)

    return light.std.MaskedMerge(strong, depth(linemask, get_depth(light)))

  def sfilt(self, light: vs.VideoNode, strong: vs.VideoNode) -> vs.VideoNode:
    return lvf.rfs(light, strong, [
        (520, 535), (561, 570), (703, 719),
        (740, 747), (865, 882), (903, 911),
        (1022, 1037), (1059, 1069), (1156, 1170),
        (1186, 1194), (1295, 1310), (1326, 1336),
        (1422, 1440), (1462, 1473), (1596, 1614),
        (1635, 1645), (1732, 1749), (1769, 1779),
        (1829, 2000)
    ])

  def rep_csharp(self, clip: vs.VideoNode) -> vs.VideoNode:
    y = get_y(clip)

    rep = core.std.Expr([
        y, y.rgsf.Repair(y, 10), y.rgsf.Repair(y, 20)
    ], 'x dup + z - x y min max x y max min')

    rep = haf.FastLineDarkenMOD(rep, 28, 3, 190, 4, 1)

    return merge_chroma(rep, clip)

  def pepegascenefilteringffs(self, clip: vs.VideoNode, deblock: vs.VideoNode, denoise: vs.VideoNode) -> vs.VideoNode:
    fix_artwork = lvf.rfs(clip, clip.std.MaskedMerge(
        deblock.std.Merge(
            denoise.resize.Bicubic(format=deblock.format.id), 0.26
        ), kgf.squaremask(clip, 697, 986, 128, 50)
    ), (2518, None))

    ms, me = (334 - 1, 355 + 1)

    return lvf.rfs(fix_artwork, fix_artwork.std.FrameEval(
        lambda n, f: clip.std.Merge(clip, (me - n) / (me - ms)), fix_artwork
    ), (ms, me))

  def grain(self, clip: vs.VideoNode) -> vs.VideoNode:
    return kgf.adaptive_grain(clip, 0.36, True, 3.5)


if __name__ == '__vapoursynth__':
  del DEBUG
  grain = Filtering().main()
  grain.set_output()
else:
  Filtering().main()
