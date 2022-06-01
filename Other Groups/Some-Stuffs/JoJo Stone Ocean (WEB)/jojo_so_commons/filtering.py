import random
import lvsfunc as lvf
import stgfunc as stg
import mvsfunc as mvf
import havsfunc as haf
import vapoursynth as vs
from functools import partial
from vardautomation import FileInfo
from vardefunc.mask import detail_mask
from vardefunc.misc import merge_chroma
from vardefunc.scale import fsrcnnx_upscale
from vardefunc.noise import Graigasm, AddGrain
from vsutil import get_y, split, depth, iterate, join
from vardefunc.util import finalise_output, initialise_input

from .scaling import Scaling
from .deblocking import Deblocking
from .utils import MMFilter, pre_aa, csharp
from .masking import mt_xxpand_multi, ExLaplaWitt

core = vs.core


class StoneOceanFiltering:
    NETFLIX: FileInfo
    BLOCKY_AF_RANGES = []

    def __init__(self, NETFLIX: FileInfo):
        self.NETFLIX = NETFLIX
        self.oycore = stg.oyster.Core()

    @finalise_output()
    @initialise_input()
    def filterchain(self, src: vs.VideoNode):
        src = src.resize.Bicubic(chromaloc_in=1, chromaloc=0)
        src_y = get_y(src)

        deb_y = haf.SMDegrain(depth(src_y, 16), 3, 250, None, True, True)
        deb_y_32 = depth(deb_y, 32)

        descale_clip = Scaling.descale(deb_y_32)

        rescaled = depth(fsrcnnx_upscale(descale_clip, shader_file=stg.misc.x56_SHADERS), 16)

        merge = Scaling.upscale_i444(merge_chroma(rescaled, src))

        deblocking = Deblocking(True)

        dpird_deblocked = [deblocking.deblock_DPIR(merge, st) for st in [30, 86]]

        deblock, deb_superhard = [
            clip.resize.Bicubic(
                format=merge.format.id, matrix=1
            )  # .fmtc.resample(sx=[0, 0.25], sy=[0, -0.075])
            for clip in dpird_deblocked
        ]

        maxm = partial(mt_xxpand_multi, M__imum=core.std.Maximum)
        minm = partial(mt_xxpand_multi, M__imum=core.std.Minimum)

        y, u, v = split(merge)

        stats = y.std.PlaneStats()

        peak = (1 << stats.format.bits_per_sample) - 1
        bin_shift = 16 << 7

        ymax, ymin = maxm(y, sw=30, mode='ellipse'), minm(y, sw=30, mode='ellipse')
        umax, umin = maxm(u, sw=2, mode='ellipse')[-1], minm(u, sw=2, mode='ellipse')[-1]
        vmax, vmin = maxm(v, sw=2, mode='ellipse')[-1], minm(v, sw=2, mode='ellipse')[-1]

        yrangesml0 = core.std.Expr([ymax[3], ymin[3]], 'x y - abs')
        yrangesml = yrangesml0.std.Binarize(2.5 * bin_shift).std.BoxBlur(0, 2, 1, 2, 1)

        rad = 16
        yrangebig0 = core.std.Expr([ymax[rad], ymin[rad]], 'x y - abs')
        yrangebig = yrangebig0.std.Binarize(4 * bin_shift)
        yrangebig = minm(yrangebig, sw=4, threshold=peak, mode='ellipse')[-4]
        yrangebig = minm(yrangebig, sw=13, threshold=(peak + 1) // 13, mode='ellipse')[-4]

        rad = 30
        ymph = core.std.Expr([
            y, maxm(ymin[rad], sw=rad, mode='ellipse')[rad],
            minm(ymax[rad], sw=rad, mode='ellipse')[rad]
        ], 'x y - z x - max')
        ymph = ymph.std.Binarize(1.5 * bin_shift).std.Minimum().std.Maximum()
        ymph = maxm(ymph, sw=5, threshold=(peak + 1) // 6)[-1]

        ymaskmain = core.std.Expr([
            yrangebig, yrangesml0.std.Binarize(560).std.BoxBlur(0, 2, 1, 2, 1)
        ], 'x y +')

        linemask = core.std.Expr(
            split(
                detail_mask(
                    merge, 50 << 8, 26 << 8, ExLaplaWitt()
                )
            ), 'x y z max max'
        ).std.Minimum().std.Deflate()

        deb_merged = deblock.std.MaskedMerge(
            merge.std.Merge(deblock.std.MaskedMerge(deb_superhard, ymaskmain), 1 / 3), linemask, 0
        )

        mask = haf.FineDehalo(deb_merged, showmask=1)

        dha = deb_merged.std.MaskedMerge(deb_superhard, mask, 0).std.Merge(deb_merged, 1 / 9)

        y_dha = get_y(dha)

        rmask = iterate(
            y_dha.std.Maximum().std.Minimum().std.Maximum().std.Binarize(1 << 10).std.Maximum(),
            core.std.Minimum, 4
        )

        emask = y_dha.std.Prewitt().std.Binarize(1 << 13).std.Maximum()
        strongmask = core.std.Expr([rmask, emask, linemask], 'y x - z +').std.Inflate()

        usml = core.std.Expr([umax, umin], 'x y - abs')
        usml = usml.std.Binarize(384).std.BoxBlur(0, 2, 1, 2, 1)
        vsml = core.std.Expr([vmax, vmin], 'x y - abs')
        vsml = vsml.std.Binarize(384).std.BoxBlur(0, 2, 1, 2, 1)

        uvmask = yrangesml.std.Expr('x 2 /')

        shitmask = join([
            yrangesml,
            core.std.Expr([usml, uvmask], 'x y +'),
            core.std.Expr([vsml, uvmask], 'x y +')
        ])

        fixed_chroma = merge_chroma(deb_merged, dha.std.MaskedMerge(
            deblock.std.MaskedMerge(
                merge.std.MaskedMerge(
                    deb_merged, shitmask
                ), strongmask
            ), uvmask, [1, 2]
        ))

        rmax = haf.mt_expand_multi(fixed_chroma, sw=1, sh=1, mode='ellipse')
        rmin = haf.mt_inpand_multi(fixed_chroma, sw=1, sh=1, mode='ellipse')

        rmix = core.std.Expr([deb_merged, rmax, rmin], 'x y min z max')

        nr = mvf.LimitFilter(rmin, rmax, thr=3 / 4, thrc=2 / 3, elast=4)

        rmix_med = core.std.Expr([rmix, rmax, nr], 'x y z min max y z max min')

        final = MMFilter([
            deblock, deb_merged, rmix_med, merge_chroma(
                csharp(
                    get_y(nr), descale_clip.resize.Bicubic(
                        rmix_med.width, rmix_med.height,
                        filter_param_a=0, filter_param_b=1
                    )
                ), pre_aa(deb_merged)
            )
        ])

        final_merge = core.average.Mean([final, merge, deb_merged, haf.FastLineDarkenMOD(final, 60)])

        freq_merge = self.oycore.FreqMerge(final_merge, deb_merged, 1)

        freq_merge = lvf.rfs(freq_merge, deb_superhard, self.BLOCKY_AF_RANGES)

        return self.graining(freq_merge)

    @initialise_input(bits=16)
    def graining(self, clip: vs.VideoNode) -> vs.VideoNode:
        seed = random.seed()

        return Graigasm(
            thrs=[x << 8 for x in (26, 87, 200)],
            strengths=[(0.88, 0.2), (0.58, 0.085), (0.4, 0.05)],
            sizes=(1.23, 1.13, 1.05),
            sharps=(80, 40, 30),
            grainers=[
                AddGrain(seed=seed, constant=False),
                AddGrain(seed=seed, constant=True),
                AddGrain(seed=seed, constant=True)
            ]
        ).graining(clip)
