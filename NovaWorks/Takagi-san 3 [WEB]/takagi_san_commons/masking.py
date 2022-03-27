import vapoursynth as vs
from lvsfunc.kernels import Catrom
from stgfunc.utils import get_bits
from vsmask.edge import PrewittStd
from typing import Optional, List, Tuple
from vsutil import split, iterate, depth, get_y

core = vs.core


def linemask(clip: vs.VideoNode) -> vs.VideoNode:
    def _linemask(plane: vs.VideoNode) -> vs.VideoNode:
        lineart_mask = PrewittStd().edgemask(plane.rgsf.RemoveGrain(3).rgsf.RemoveGrain(3).rgsf.RemoveGrain(3), 0.1)
        lineart_mask = lineart_mask.std.Maximum().std.Minimum().std.Minimum().std.Deflate().std.Deflate()
        lineart_mask = lineart_mask.std.Expr(
            'x 7.6 *').rgsf.RemoveGrain(5).rgsf.RemoveGrain(5).std.Maximum().std.Minimum()
        lineart_mask = lineart_mask.std.Minimum().std.Maximum().std.Minimum().std.Maximum().std.Minimum().std.Maximum()
        lineart_mask = lineart_mask.std.Minimum().std.Maximum().std.Minimum().std.Maximum().std.Minimum().std.Maximum()
        return lineart_mask.std.Minimum().std.Maximum().std.Minimum().std.Binarize(116 << 8)

    return core.std.Expr([
        _linemask(p) for p in split(Catrom().resample(clip, vs.RGB48).std.RemoveFrameProps('_Matrix'))
    ], 'x y z + +')


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
