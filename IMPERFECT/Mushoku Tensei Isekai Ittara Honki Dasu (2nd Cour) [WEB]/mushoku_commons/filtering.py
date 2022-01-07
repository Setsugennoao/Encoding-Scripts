import EoEfunc as eoe
import lvsfunc as lvf
import stgfunc as stg
import havsfunc as hvf
import vapoursynth as vs
from stgfunc import depth
from vsutil import get_y, split
from vardefunc.noise import decsiz
from lvsfunc.kernels import Bicubic
from vardautomation import FileInfo
from vardefunc.deband import dumb3kdb
from vardefunc.misc import merge_chroma
from vardefunc.util import finalise_output
from vardefunc.mask import ExLaplacian4, PrewittStd, region_mask, detail_mask
from typing import Optional, Tuple


core = vs.core


class ExLaplaWitt(ExLaplacian4):
  def _compute_mask(self, clip: vs.VideoNode) -> vs.VideoNode:
    exlaplacian4 = super()._compute_mask(clip)
    prewitt = PrewittStd().get_mask(clip)
    mask = core.std.Expr((exlaplacian4, prewitt), 'x y max')
    return region_mask(mask, right=2).fb.FillBorders(right=2)


class MushokuFiltering:
  def __init__(self, FUNI: FileInfo, den_strength: Optional[Tuple[Optional[float], Optional[float]]] = None):
    self.FUNI = FUNI
    self.denoise_strength = den_strength or [None, 0.56]

  @finalise_output()
  def filterchain(self) -> vs.VideoNode:
    src = depth(self.FUNI.clip_cut, 16)

    luma = get_y(src)

    kernel = Bicubic()

    luma32 = depth(luma, 32)

    descale = kernel.descale(luma32, 1500, 844)
    rescale = kernel.scale(descale, 1920, 1080)

    descale_mask = lvf.scale.descale_detail_mask(luma32, rescale)

    upscale = stg.upscale.upscale(descale)

    upscale = upscale.std.MaskedMerge(luma32, descale_mask)

    upscale = depth(upscale, 16)

    merged = merge_chroma(upscale, src)

    pre = hvf.SMDegrain(merged, tr=2, thSADC=300, plane=3)

    denoise = eoe.dn.BM3D(stg.denoise.KNLMeansCL(pre, sigma=self.denoise_strength), 1.25, chroma=False)

    preden = core.dfttest.DFTTest(denoise, sbsize=16, sosize=12, tbsize=1)
    detailmask = core.std.Expr(
        split(
            detail_mask(
                preden, brz_mm=2500, brz_ed=1400,
                edgedetect=ExLaplaWitt()
            ).resize.Bilinear(format=vs.YUV444P16)
        ), 'x y z max max'
    )

    deband = dumb3kdb(preden, 16, 30, grain=[24, 0])
    deband = core.std.MergeDiff(deband, denoise.std.MakeDiff(preden))
    deband = core.std.MaskedMerge(deband, denoise, detailmask)

    return decsiz(deband, min_in=128 << 8, max_in=192 << 8)
