from functools import partial
from typing import Any, List

import lvsfunc as lvf
import stgfunc as stg
import vapoursynth as vs
from debandshit import dumb3kdb, f3kbilateral, placebo_deband
from stgfunc import Grainer, SetsuCubic, adaptive_grain
from vardautomation import get_vs_core
from vardefunc import decsiz, finalise_clip, fsrcnnx_upscale, merge_chroma
from vsdehalo import HQDeringmod, edge_cleaner, fine_dehalo
from vsdenoise import BM3DCudaRTC, CCDMode, CCDPoints, ChannelMode, MVTools, Prefilter, Profile, ccd, knl_means_cl
from vskernels import Catrom, Mitchell, Robidoux
from vsmask.edge import FDoGTCanny, Kirsch
from vsrgtools import contrasharpening, contrasharpening_dehalo, lehmer_diff_merge
from vsutil import depth, get_peak_value, get_y

from .utils import EPS_ED_RANGES, EPS_OP_RANGES, EPS_SOURCES, merge_episodes

core = get_vs_core(range(0, vs.core.num_threads, 2))
catrom = Catrom()


def filterchain(
    idx: int,
    MEDIUM_GRAIN_BUT_IDK_MAN_THE_MOTION_BLOCKS_ARE_DYING: List[lvf.types.Range] = [],
    SUPER_COARSE_GRAINY_WTF_KILL_THIS_STUDIO_PLEASE_RANGES: List[lvf.types.Range] = [],
    VSDPIR_DEBLOCK_RANGES_JESUSSSSS: List[lvf.types.Range] = [],
    EPIC_DEBANDING_RANGES: List[lvf.types.Range] = [],
    BIG_ASS_GRAIN_DUDE_PLEASE_CMON: List[lvf.types.Range] = [],
    cour: int = 1
) -> vs.VideoNode:
    cour -= 1
    idx -= 12 * cour

    src = EPS_SOURCES[cour][idx]
    OP_RANGES = EPS_OP_RANGES[cour][idx]
    ED_RANGES = EPS_ED_RANGES[cour][idx]

    eps_OPED_average = merge_episodes(idx, cour)

    vinv = lvf.vinverse(src, 2, 6, 0.85)
    vinv = lvf.rfs(vinv, eps_OPED_average, OP_RANGES)

    vinv_y = get_y(vinv)

    planestats = get_y(src).std.PlaneStats()

    peak = get_peak_value(planestats)

    maskstats0, maskstats1 = planestats.adg.Mask(9.5), planestats.adg.Mask(13.5)

    kirsch, fdog = Kirsch().edgemask(planestats), FDoGTCanny().edgemask(planestats)

    mask = core.akarin.Expr(
        [maskstats0, maskstats1, kirsch, fdog],
        f'y x - z + x 2 / + y - X! {peak} P! P@ P@ X@ - / L! 1 L@ - X@ * x L@ * +'
    )

    mask = mask.bilateral.Gaussian(None, 1.5)
    mask = mask.std.MakeDiff(mask).std.Expr(f'x {peak} 2 / -')
    mask = mask.std.Binarize(1).bilateral.Gaussian(1)

    tden_chroma = ccd(vinv, 3.5)
    tdenoise = get_y(tden_chroma)

    bmdenoise = BM3DCudaRTC(tdenoise, [1.5, 0], 1, Profile.NORMAL).clip

    tdenoise = contrasharpening(bmdenoise, vinv_y).std.MaskedMerge(bmdenoise, fdog)

    denoise = knl_means_cl(tdenoise, 1.15, 2)

    mv = MVTools(merge_chroma(denoise, src), 1)
    mv.analyze()

    schizo_degrain = BM3DCudaRTC(mv.degrain(None, 480), 0.85).clip

    halo_mask = fine_dehalo(tdenoise, show_mask=True).std.Maximum()

    denoise_contra = contrasharpening_dehalo(
        denoise, vinv_y, 1.6
    ).std.MaskedMerge(
        tdenoise, core.std.Expr([mask, kirsch], 'x y -')
    ).std.MaskedMerge(
        bmdenoise, halo_mask
    )

    rescale = fsrcnnx_upscale(
        depth(
            SetsuCubic().descale(depth(denoise_contra, 32), 1685, 948), 16
        ), 1920, 1080, stg.misc.x56_SHADERS, strength=75
    )

    denoise_rescale = contrasharpening_dehalo(
        core.std.Expr([
            denoise_contra,
            fsrcnnx_upscale(
                depth(
                    Mitchell().descale(
                        depth(denoise_contra, 32), 1600, 900
                    ).resize.Bicubic(1685, 948).std.Merge(
                        Robidoux().descale(
                            depth(denoise_contra, 32), 1280, 720
                        ).resize.Bicubic(1685, 948)
                    ), 16
                ), 1920, 1080, stg.misc.x56_SHADERS, strength=85
            ).std.MaskedMerge(denoise_contra, fdog.std.Inflate().std.Invert().std.Deflate())
        ], 'x y min'), bmdenoise, 2
    )

    denoise_rescale = contrasharpening(denoise_rescale, tdenoise)
    denoise_rescale = HQDeringmod(
        denoise_rescale, Prefilter.GAUSSBLUR2, mrad=2, contra=False
    )

    denoise_rescale = lehmer_diff_merge(denoise_rescale, rescale, partial(
        Prefilter.DFTTEST, sbsize=9, smode=0, sosize=0, tosize=0, tmode=0,
        slocation=[0.0, 0.0, 0.12, 1024.0, 1.0, 1024.0]
    ))

    denoise_rescale = edge_cleaner(
        rescale.std.MaskedMerge(
            contrasharpening_dehalo(rescale, denoise_rescale, 1.5), fdog
        ), 12.5, 17, True, True
    )

    aamerge = merge_chroma(denoise_rescale, tden_chroma)
    based_aa = aamerge.std.MaskedMerge(
        knl_means_cl(aamerge, 1.65, channels=ChannelMode.CHROMA),
        core.std.Expr([halo_mask, fdog], 'x y max').std.Deflate()
    )
    aa = ccd(based_aa, 2.65, mode=CCDMode.NNEDI_BICUBIC, scale=1.25, ref_points=CCDPoints.ALL)
    aa = lvf.rfs(aa, tden_chroma, OP_RANGES)
    aa = lvf.rfs(aa, merge_chroma(bmdenoise, tden_chroma), ED_RANGES)

    aa = knl_means_cl(aa, 0.38, channels=ChannelMode.CHROMA).std.MaskedMerge(based_aa, fdog, [1, 2])

    byaa = get_y(based_aa)

    aa = merge_chroma(
        contrasharpening_dehalo(get_y(aa), tdenoise, 1).std.MaskedMerge(byaa, fdog.std.Inflate()), aa
    )

    dpir_kwargs = dict[str, Any]()

    if len(VSDPIR_DEBLOCK_RANGES_JESUSSSSS):
        dpir_kwargs |= dict(zones=[
            (VSDPIR_DEBLOCK_RANGES_JESUSSSSS, 30)
        ])

    deblock = lvf.dpir(aa, 14, cuda='trt', **dpir_kwargs)

    deband = dumb3kdb(deblock, 16, 32, 0)

    deband = lvf.rfs(
        deband, merge_chroma(deband, ccd(vinv, 0.5, 1, ref_points=CCDPoints.HIGH)), BIG_ASS_GRAIN_DUDE_PLEASE_CMON
    )

    if EPIC_DEBANDING_RANGES:
        hard_deband = placebo_deband(f3kbilateral(deband, 18, 48, 2))
        deband = lvf.rfs(deband, hard_deband, EPIC_DEBANDING_RANGES)

    grain = adaptive_grain(
        deband, [0.1, 0.06], 0.95, 65, False, 10, Grainer.AddNoise, temporal_average=2
    )

    if cour == 1:
        grain = lvf.rfs(
            grain,
            grain.std.MaskedMerge(
                grain.grain.Add(2.6, 0.25, seed=6969, constant=False), mask
            ).grain.Add(1.85, 1.3, seed=42069, constant=False),
            OP_RANGES
        )

        assert grain.format

        edmask = stg.src(r".\masks\edmask.png", matrix_prop=1, ref=deband).resize.Bicubic(format=vs.GRAY16)
        edmask = edmask.bilateral.Gaussian(15).std.Crop(0, 0, 137, 137).std.AddBorders(40, 40, 240, 240)
        edmask = edmask.resize.Bicubic(1920, 1080).std.AddBorders(60, 60)
        edmask = edmask.bilateral.Gaussian(25, 50).std.Crop(60, 60, 137, 137).std.AddBorders(0, 0, 137, 137)

        grain = lvf.rfs(
            grain,
            grain
            .resize.Bicubic(format=vs.RGBS, matrix_in=1)
            .std.MaskedMerge(
                grain.std.Merge(schizo_degrain)
                .resize.Bicubic(format=vs.RGBS, matrix_in=1)
                .grain.Add(4.5, 2.12, seed=6330, constant=False),
                depth(edmask, 32)
            )
            .std.Crop(0, 0, 137, 137)
            .std.AddBorders(0, 0, 137, 137)
            .resize.Bicubic(format=grain.format.id, matrix=1),
            ED_RANGES
        )
    else:
        grain = lvf.rfs(grain, denoise, ED_RANGES)

    grain = lvf.rfs(
        grain,
        schizo_degrain
        .grain.Add(2.4, 0.75, seed=2356, constant=False)
        .grain.Add(2.8, 0.5, seed=123456, constant=False),
        MEDIUM_GRAIN_BUT_IDK_MAN_THE_MOTION_BLOCKS_ARE_DYING
    )

    grain = lvf.rfs(
        grain,
        schizo_degrain
        .grain.Add(1.85, 0.25, seed=7896, constant=False)
        .grain.Add(4.2, 1.3, seed=9999, constant=False),
        SUPER_COARSE_GRAINY_WTF_KILL_THIS_STUDIO_PLEASE_RANGES
    )

    grain = lvf.rfs(
        grain,
        adaptive_grain(
            deband, [0.15, 0.25], 4.5, 100, False, 5.5, Grainer.AddNoise
        ),
        BIG_ASS_GRAIN_DUDE_PLEASE_CMON
    )

    grain = decsiz(grain, min_in=128 << 8, max_in=200 << 8)

    return finalise_clip(grain)
