from functools import partial
from typing import Any, List

import lvsfunc as lvf
import stgfunc as stg
import vapoursynth as vs
from debandshit import dumb3kdb, f3kbilateral, placebo_deband
from stgfunc import Grainer, SetsuCubic, adaptive_grain
from vardautomation import get_vs_core
from vardefunc import decsiz, finalise_clip, fsrcnnx_upscale, merge_chroma
from vsdehalo import edge_cleaner, fine_dehalo
from vsdenoise import BM3DCudaRTC, CCDMode, CCDPoints, ChannelMode, MVTools, Prefilter, Profile, ccd, knl_means_cl
from vskernels import Catrom, Mitchell, Robidoux
from vsmask.edge import FDoGTCanny, Kirsch
from vsrgtools import contrasharpening, contrasharpening_dehalo, lehmer_diff_merge, gauss_blur
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

    planestats = get_y(src).std.PlaneStats()

    peak = get_peak_value(planestats)

    maskstats0, maskstats1 = planestats.adg.Mask(9.5), planestats.adg.Mask(13.5)

    kirsch, fdog = Kirsch().edgemask(planestats), FDoGTCanny().edgemask(planestats)
    fdog_inflate = fdog.std.Inflate()

    mask = core.akarin.Expr(
        [maskstats0, maskstats1, kirsch, fdog],
        f'y x - z + x 2 / + y - X! {peak} P! P@ P@ X@ - / L! 1 L@ - X@ * x L@ * +'
    )

    blur_mask = gauss_blur(mask, 1.5)
    mask = mask.std.MakeDiff(blur_mask).std.Expr(f'x {peak} 2 / -')
    mask = gauss_blur(mask.std.Binarize(1), 1)

    tdenoise = ccd(vinv, 3.5)

    bmdenoise = BM3DCudaRTC(tdenoise, [1.5, 0], 1, Profile.NORMAL).clip

    tdenoise = contrasharpening(bmdenoise, vinv).std.MaskedMerge(bmdenoise, fdog, 0)

    denoise = knl_means_cl(tdenoise, 0.95, 1, channels=ChannelMode.LUMA)

    mv = MVTools(denoise, 1)
    mv.analyze()

    schizo_degrain = BM3DCudaRTC(mv.degrain(None, 480), 0.85).clip

    halo_mask = fine_dehalo(tdenoise, show_mask=True).std.Maximum()

    denoise_contra = contrasharpening_dehalo(
        denoise, vinv, 1.6
    ).std.MaskedMerge(
        tdenoise, core.std.Expr([mask, kirsch], 'x y -')
    ).std.MaskedMerge(
        bmdenoise, halo_mask
    )

    contra_y = get_y(denoise_contra)
    contra32 = depth(contra_y, 32)

    setsu_descale = SetsuCubic().descale(contra32, 1685, 948)
    mitch_descale = Mitchell().descale(contra32, 1600, 900)
    roubi_descale = Robidoux().descale(contra32, 1280, 720)

    shit_avg_clips = [c.resize.Bicubic(1685, 948) for c in (mitch_descale, roubi_descale)]

    shit_avg = core.std.Expr(shit_avg_clips, 'x y + 32768 *', vs.GRAY16)
    rescale_mask = fdog_inflate.std.Invert().std.Deflate()

    rescale = fsrcnnx_upscale(
        depth(setsu_descale, 16), 1920, 1080, stg.misc.x56_SHADERS, strength=75
    )
    shit_rescale = fsrcnnx_upscale(
        shit_avg, 1920, 1080, stg.misc.x56_SHADERS, strength=85
    )
    shit_rescale = shit_rescale.std.MaskedMerge(contra_y, rescale_mask)

    rescaled = core.std.Expr([contra_y, shit_rescale], 'x y min')

    denoise_rescale = contrasharpening_dehalo(rescaled, get_y(bmdenoise), 2)

    denoise_rescale = lehmer_diff_merge(denoise_rescale, rescale, high_filter=partial(
        Prefilter.DFTTEST, sbsize=9, smode=0, sosize=0, tosize=0, tmode=0,
        slocation=[0.0, 0.0, 0.12, 1024.0, 1.0, 1024.0]
    ))
    denoise_rescale = contrasharpening(denoise_rescale, get_y(tdenoise))

    denoise_rescale = edge_cleaner(
        rescale.std.MaskedMerge(
            contrasharpening_dehalo(rescale, denoise_rescale, 1.5), fdog
        ), 12.5, 17, True, True
    )

    halo_mask_hat = core.std.Expr([halo_mask, fdog], 'x y max').std.Deflate()

    knl_cdenoise = knl_means_cl(tdenoise, 1.55, channels=ChannelMode.CHROMA)

    based_cdenoise = tdenoise.std.MaskedMerge(knl_cdenoise, halo_mask_hat, [1, 2])
    cdenoise = ccd(
        based_cdenoise, 2.65, mode=CCDMode.NNEDI_BICUBIC, scale=1.25, ref_points=CCDPoints.ALL
    )
    cdenoise = merge_chroma(denoise_rescale, cdenoise)
    cdenoise = lvf.rfs(cdenoise, tdenoise, OP_RANGES)
    cdenoise = lvf.rfs(cdenoise, bmdenoise, ED_RANGES)

    cdenoise = knl_means_cl(
        cdenoise, 0.38, channels=ChannelMode.CHROMA
    ).std.MaskedMerge(based_cdenoise, fdog, [1, 2])

    cdenoise = contrasharpening_dehalo(cdenoise, tdenoise, 1).std.MaskedMerge(based_cdenoise, fdog_inflate, 0)

    dpir_kwargs = dict[str, Any]()

    if len(VSDPIR_DEBLOCK_RANGES_JESUSSSSS):
        dpir_kwargs |= dict(zones=[
            (VSDPIR_DEBLOCK_RANGES_JESUSSSSS, 30)
        ])

    deblock = lvf.dpir(cdenoise, 13.65, cuda='trt', **dpir_kwargs)

    deband = dumb3kdb(deblock, 16, 32, 0)

    deband = lvf.rfs(
        deband, merge_chroma(
            deband, ccd(vinv, 0.5, 1, ref_points=CCDPoints.HIGH, planes=[1, 2])
        ), BIG_ASS_GRAIN_DUDE_PLEASE_CMON
    )

    if EPIC_DEBANDING_RANGES:
        hard_deband = placebo_deband(f3kbilateral(deband, 18, 48, 2))
        deband = lvf.rfs(deband, hard_deband, EPIC_DEBANDING_RANGES)

    grain = adaptive_grain(
        deband, [0.1, 0.06], 0.95, 65, False, 10, Grainer.AddNoise, temporal_average=2
    )

    if cour == 0:
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
