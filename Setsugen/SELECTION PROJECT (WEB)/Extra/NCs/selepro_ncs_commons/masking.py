import vardefunc as vdf
import vapoursynth as vs
from vsutil import get_y, depth, iterate
from typing import Optional, Tuple, List

from .utils import get_bits


core = vs.core

# Stolen from Light <3


def detail_mask(
        clip: vs.VideoNode,
        sigma: float = 1.0, rxsigma: List[int] = [50, 200, 350],
        pf_sigma: Optional[float] = 1.0, brz: Tuple[int, int] = (2500, 4500),
        rg_mode: int = 17
) -> vs.VideoNode:
  bits, clip = get_bits(clip)

  clip_y = get_y(clip)
  pf = core.bilateral.Gaussian(clip_y, sigma=pf_sigma) if pf_sigma else clip_y
  ret = core.retinex.MSRCP(pf, sigma=rxsigma, upper_thr=0.005)

  blur_ret = core.bilateral.Gaussian(ret, sigma=sigma)
  blur_ret_diff = core.std.Expr([blur_ret, ret], "x y -")
  blur_ret_dfl = core.std.Deflate(blur_ret_diff)
  blur_ret_ifl = iterate(blur_ret_dfl, core.std.Inflate, 4)
  blur_ret_brz = core.std.Binarize(blur_ret_ifl, brz[0])
  blur_ret_brz = core.morpho.Close(blur_ret_brz, size=8)

  prewitt_mask = core.std.Prewitt(clip_y).std.Binarize(brz[1])
  prewitt_ifl = prewitt_mask.std.Deflate().std.Inflate()
  prewitt_brz = core.std.Binarize(prewitt_ifl, brz[1])
  prewitt_brz = core.morpho.Close(prewitt_brz, size=4)

  merged = core.std.Expr([blur_ret_brz, prewitt_brz], "x y +")
  rm_grain = core.rgvs.RemoveGrain(merged, rg_mode)
  return rm_grain if bits == 16 else depth(rm_grain, bits)
