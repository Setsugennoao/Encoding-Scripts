from ccd import ccd
import lvsfunc as lvf
import stgfunc as stg
import vapoursynth as vs
from functools import partial
from math import asinh, acos, cos
from vsmask.edge import PrewittStd
from rgvs import removegrain, repair
from stgfunc.deband import auto_deband
from stgfunc.tweaking import bbmod_fast
from vardefunc.misc import YUVPlanes, merge_chroma
from stgfunc.easing import SineEaseIn, QuadEaseInOut
from vsdenoise import BM3DCuda, Profile, knl_means_cl
from fine_dehalo import fine_dehalo, contrasharpening
from vardautomation import X265, FileInfo, get_vs_core
from vardefunc.util import finalise_clip, initialise_clip
from debandshit import placebo_deband, f3kbilateral, f3kpf
from stgfunc.transitions import panner, PanFunction, PanDirection
from vardefunc.aa import Eedi3SS, Nnedi3SS, Eedi3SR, upscaled_sraa
from vardefunc.noise import Graigasm, AddGrain, decsiz, BilateralMethod
from lvsfunc.kernels import Catrom, BicubicDidee, Bicubic, Lanczos, RobidouxSoft
from vsutil import depth, Dither, get_peak_value, get_neutral_value, get_y, join, insert_clip
from stgfunc.utils import combine, ExprOp, pad_reflect, squaremask, replace_squaremask, freeze_replace_mask


core = get_vs_core(range(0, vs.core.num_threads, 2))

is_debug = __name__ != '__main__'

catrom = Catrom()

didee = BicubicDidee()
robisoft = RobidouxSoft()
pompobic = Bicubic(
    asinh(.5) * acos(.5) * cos(400),
    abs(asinh(.5) * acos(-.5) * cos(450))
)


# 31261, 40162, 40274, 40558, 40600
# chroma 27052, 30503
# floor 27052, 30503
# cardbord cutout 27113, 32279, 33576, 41379
# transition 70190
# dehardsub
#   26261, 93815, 95383, 95865, 116049,
#   127996, 128207, 128320, 128383,
#   128453, 128794
# custom antialiasing
#   (17432, 17541), (79141, 79206) 19598,
#   20872, 25922, (33193, 33246), 41659
# 54877 broken coloring
# aaand many more :(

# awful zoom 1510 :thumbsup:

aspect_ratio_21_9_ranges = [
    (7735,   7866), (9036,   9329), (9374,   9797),
    (9885,  10004), (17010, 17431), (52741, 52808),
    (52941, 53012), (53073, 53362), (57043, 57824),
    (59038, 59432), (60639, 61169), (61229, 61991),
    (62184, 62351), (62718, 62813), (62874, 63012),
    (63164, 63772), (70213, 70374), (72839, 73343),
    (73554, 73771), (74020, 75086), (75334, 75708),
    (76083, 76230), (77822, 77869), (78264, 78382),
    (78425, 78640), (78725, 78817), (78997, 79140),
    (79345, 79814), (79893, 80402), (80535, 80847),
    (80908, 81044), (81377, 81454), (85184, 85211),
    (85247, 85548), (85665, 85891), (86143, 86598),
    (86809, 87638), (87868, 87909), (87951, 87995),
    (90520, 91564), (91697, 91720), (91745, 91780),
    (91823, 91836), (91885, 91911), (92515, 92607),
    (115887, 117471), (120757, 120798), (120904, 120935),
    (120963, 121038), (122768, 122779), (122802, 122813),
    (123822, 125510), (125589, 126062)
]

untouched_1080p_scenes_ranges = [
    (27620, 27916), (29644, 29712), (30629, 30684),
    (32272, 32274), (41395, 41398), (48635, 48694),
    (50494, 50656), (58309, 58356), (71680, 71793),
    (72711, 72838), (73492, 73537), (73862, 73949),
    (75087, 75133), (75278, 75297), (75709, 75840),
    (76007, 76082), (76567, 76715), (77870, 77911),
    (78014, 78085), (78401, 78424), (78641, 78670),
    (78818, 78853), (78902, 78943), (79285, 79308),
    (79815, 79856), (80499, 80534), (80872, 80907),
    (85212, 85246), (87928, 87950), (99398, 99431),
    (99468, 99764), (100135, 100242), (100375, 100522),
    (119274, 119348), (119457, 119501), (120457, 120653),
    (120694, 120756), (121548, 121568), (121904, 121999),
    (122412, 122426), (123452, 123545), (129590, 129631),
    (129642, None)
]

banding_galore_ranges = [
    (5779,   6077), (6254,   6658), (6876,   7102),
    (7698,   7745), (7854,   7909), (9036,   9059),
    (9681,   9697), (11273, 11387), (12826, 13233),
    (17432, 17540), (20238, 20424), (22342, 22433),
    (23146, 23773), (23828, 24085), (24577, 24857),
    (25360, 25417), (25463, 25549), (27336, 27515),
    (30511, 30606), (31147, 31261), (32312, 32316),
    (32320, 32358), (32812, 32852), (37257, 37259),
    (47025, 47195), (48917, 49103), (52730, 52808),
    (56997, 57751), (62352, 62387), (70375, 70403),
    (73772, 73861), (79057, 79140), (122225, 122264),
    (129650, 129654)
]

no_descale_ranges = [
    (2392,   2405), (2993,   3060), (3678,   3816),
    (4525,   4704), (9798,   9884), (19333, 19458),
    (25463, 25549), (31147, 31261), (35144, 35281),
    (63013, 63054), (45731, 45826), (52725, 52740),
    (69096, 69233), (70195, 70374), (70443, 70515),
    (72887, 72934), (86635, 86760), (101370, 101477),
    (114421, 114468), (121290, 121436), (121569, 121807),
    (122265, 122311), (122339, 122358), (122375, 122387),
    (122427, 122767)
]

stronger_aa_ranges = [
    (19566, 19650), (20868, 20919), (41652, 41692),
    (33193, 33246), (33544, 33595), (84882, 84953),
    (88753, 88844), (89345, 89494), (91565, 91612),
    (95067, 95261)
]

nuke_aa_ranges = [
    (27096, 27263), (32279, 32281), (41379, 41382)
]

twenty_tons_nuke_ranges = [
    (34702, 34761), (37206, 37256), (52448, 52494),
    (114775, 114864)
]

idk_chief_deband_ranges = [
    (0, 7), (129638, 129641), (129644, 129649),
    (129655, 129659)
]

light_debanding_ranges = [
    (35144, 35281), (46188, 46228)
]

hard_denoising_ranges = [
    (0, 13), (34702, 34761), (73772, 73861), (79057, 79140),
    (114775, 114864)
]

broken_border_ranges = [
    (1916, 1989)
]

onscreen_text_1080p_ranges = [
    (1714, 1769), (1864, 1982), (1990, 2097),
    (2128, 2207), (2412, 2490), (2534, 2608),
    (2653, 2753), (2761, 2897), (2935, 2992),
    (3200, 3288), (34062, 34217)
]

only_dehalo_1080p_ranges = [
    1510, (1511, 1576), (1619, 1678), (27336, 27547),
    (34762, 34905), (43008, 43193), (46277, 46325)
]

no_aa_ranges = [
    (72291, 72384)
]

no_debanding_ranges = [
    (48049, 48145)
]

deblocking_ranges = [
    (2404, 2612), (2761, 2897)
]

pepega_movie_end_range = (129660, None)

JPBD = FileInfo(
    r"E:\Desktop\Encoding Sources\BDMVs - ISOs\Eiga Daisuki Pompo-san JP BDMV\BIXA_1314\BDMV\STREAM\00004.m2ts",
    (660, -42), idx=partial(stg.src, force_lsmas=True)
)

src = initialise_clip(JPBD.clip_cut)

peak = get_peak_value(src)
neutral = get_neutral_value(src)

crop_catrom = Catrom(src_height=1036)

with YUVPlanes(src) as c:
    c.Y = crop_catrom.scale(c.Y, 1920, 1036, (21, 0))
    c.U = crop_catrom.scale(
        c.U.std.Transpose().nnedi3.nnedi3(1, 1, 0, 0, 3, 1).std.Transpose()
        .nnedi3.nnedi3(0, 1, 0, 0, 3, 1), 1920, 1036, (21.5, 0)
    )
    c.V = crop_catrom.scale(
        c.V.std.Transpose().nnedi3.nnedi3(1, 1, 0, 0, 3, 1).std.Transpose()
        .nnedi3.nnedi3(0, 1, 0, 0, 3, 1), 1920, 1036, (21.5, 0)
    )


cropped = c.clip

cropped_21_9 = pad_reflect(cropped.std.Crop(0, 0, 106, 102), 0, 0, 104, 104)

crop = lvf.rfs(cropped, cropped_21_9, aspect_ratio_21_9_ranges)

edgefix = bbmod_fast(crop, 2, 2)
strong_ef = bbmod_fast(edgefix, 4, 3)

edgefix = lvf.rfs(edgefix, strong_ef, broken_border_ranges)


def fuck_aliasing(clip: vs.VideoNode) -> vs.VideoNode:
    src = clip[1250:1320]
    shifted = src[1:].resize.Bicubic(src_left=-0.55) + src[-2].resize.Bicubic(src_left=0.55)
    shifted1 = src[0].resize.Bicubic(src_left=-0.55) + src[:-1].resize.Bicubic(src_left=0.55)

    fixed = core.std.Expr(
        [src, shifted, shifted1], 'x y z max max'
    ).eedi3m.EEDI3CL(
        1, 0, [0, 1, 2], 0.15, 0.25, 160, 3, 15
    )

    mask_frames = [
        core.std.Expr([
            squaremask(src, 30, 490, 863, 0),
            squaremask(src, 20, 497, 982, 0),
            squaremask(src, 20, 500, 1090, 0),
            fixed.std.ShufflePlanes(0, vs.GRAY).std.Sobel()
        ], 'x y z max max 0 = 0 a 10 * ?').resize.Bicubic(src_left=-1)[2]
    ] * fixed.num_frames

    mask_frames[0:2] = [*[
        core.std.Expr([
            squaremask(src, 30, 450, 863, 0),
            squaremask(src, 20, 490, 982, 0),
            squaremask(src, 20, 500, 1090, 0),
            fixed.std.ShufflePlanes(0, vs.GRAY).std.Sobel()
        ], 'x y z max max 0 = 0 a 10 * ?').resize.Bicubic(src_left=-1)[2]
    ] * fixed.num_frames][0:2]
    mask_frames[2] = core.std.Expr([
        squaremask(src, 30, 452, 863, 0),
        squaremask(src, 20, 475, 982, 0),
        squaremask(src, 20, 500, 1090, 0),
        fixed.std.ShufflePlanes(0, vs.GRAY).std.Sobel()
    ], 'x y z max max 0 = 0 a 10 * ?').resize.Bicubic(src_left=-1)[2]

    mask_frames[3:5] = [*[
        core.std.Expr([
            squaremask(src, 30, 490, 863, 0),
            squaremask(src, 20, 475, 982, 0),
            squaremask(src, 20, 500, 1090, 0),
            fixed.std.ShufflePlanes(0, vs.GRAY).std.Sobel()
        ], 'x y z max max 0 = 0 a 10 * ?').resize.Bicubic(src_left=-1)[2]
    ] * fixed.num_frames][3:5]

    mask_frames[10:21] = [*[
        core.std.Expr([
            squaremask(src, 30, 490, 863, 0),
            squaremask(src, 20, 427, 982, 0),
            squaremask(src, 20, 500, 1090, 0),
            fixed.std.ShufflePlanes(0, vs.GRAY).std.Sobel()
        ], 'x y z max max 0 = 0 a 10 * ?').resize.Bicubic(src_left=-1)[2]
    ] * fixed.num_frames][10:21]
    mask_frames[32:41] = [*[
        core.std.Expr([
            squaremask(src, 30, 490, 863, 0),
            squaremask(src, 20, 490, 982, 0),
            squaremask(src, 20, 443, 1090, 0),
            fixed.std.ShufflePlanes(0, vs.GRAY).std.Sobel()
        ], 'x y z max max 0 = 0 a 10 * ?').resize.Bicubic(src_left=-1)[2]
    ] * fixed.num_frames][32:41]
    mask_frames[49:60] = [*[
        core.std.Expr([
            squaremask(src, 30, 490, 863, 0),
            squaremask(src, 20, 432, 982, 0),
            squaremask(src, 20, 500, 1090, 0),
            fixed.std.ShufflePlanes(0, vs.GRAY).std.Sobel()
        ], 'x y z max max 0 = 0 a 10 * ?').resize.Bicubic(src_left=-1)[2]
    ] * fixed.num_frames][49:60]

    mask_shifting = core.std.Splice([
        frame.resize.Bicubic(src_left=0.6 * i)
        for i, frame in enumerate(mask_frames, -1)
    ])

    mix = src.std.MaskedMerge(fixed, mask_shifting)

    # all this shit and i still couldn't fix it properly
    clip = insert_clip(clip, mix, 1250)

    clip = lvf.rfs(
        clip, replace_squaremask(
            lvf.sraa(clip, 1.85), clip,
            (660, 590, 775, 445), blur_sigma=15
        ), (17432, 17540)
    )

    clip = lvf.rfs(
        clip, core.std.Expr([
            lvf.sraa(clip, 1.65), lvf.sraa(clip, 1.35)
        ], 'x y min'), stronger_aa_ranges
    )

    manual_mask_79141 = stg.src(r".\assets\manual_mask_79141.png").resize.Bicubic(format=vs.YUV444P16, matrix=1)

    manual_mask_79141 = get_y(manual_mask_79141).std.Binarize(235 << 8).bilateral.Gaussian(2.5)

    clip = lvf.rfs(
        clip, clip.std.MaskedMerge(
            lvf.sraa(
                lvf.sraa(clip.std.Transpose(), 1.35).std.Transpose(), 1.95
            ), manual_mask_79141
        ), (79141, 79206)
    )

    manual_mask_71905 = stg.src(r".\assets\manual_mask_71905.png",).resize.Bicubic(format=vs.YUV444P16, matrix=1)

    manual_mask_71905 = get_y(manual_mask_71905).std.Binarize(235 << 8) \
        .std.Maximum().std.Maximum().bilateral.Gaussian(2.5).resize.Bicubic(src_left=-4.5, src_top=-4.25)

    clip = lvf.rfs(
        clip, clip.std.MaskedMerge(
            clip.bilateral.Gaussian(3), manual_mask_71905
        ).std.MaskedMerge(
            clip, manual_mask_71905.std.Binarize(16 << 8)
            .std.InvertMask().std.Maximum().bilateral.Gaussian(1).resize.Bicubic(src_top=4.55, src_left=-1)
        ), (71905, 72201)
    )

    y = get_y(clip)

    eedi_supersampler = Eedi3SS(
        True, False, 0.2, 0.65, 820, 1, 25
    )

    eedisr2 = Eedi3SR(
        True, False, 0.25, gamma=160
    )

    eedisr1 = Eedi3SR(
        True, False, 0.35, gamma=180
    )

    eedisr3 = Eedi3SR(
        True, False, 0.2, 0.6, 20, 3, 35
    )

    fix = upscaled_sraa(
        upscaled_sraa(
            eedisr2.aa(
                eedisr1.aa(
                    y.std.Transpose()
                ).std.Transpose()
            ), 1.85, supersampler=eedi_supersampler,
            singlerater=eedisr3
        ), 1.75
    )

    fix2 = stg.oyster.Core().FreqMerge(fix, y, 15).std.Merge(fix, 5 / 17)
    fix3 = merge_chroma(upscaled_sraa(fix2, 1.65), clip)

    clip = lvf.rfs(clip, fix3, nuke_aa_ranges)

    return clip


edgefix = fuck_aliasing(edgefix)

rg11 = core.akarin.Expr([
    edgefix.akarin.Expr(f'{neutral} n! x x 2 * x[0,-1] x[0,1] + + 4 / - n@ +'), edgefix
], f"""
    {neutral} n! x 2 * x[0,-1] x[0,1] + + 4 / blur! y x blur@ - x n@ - * 0 < n@ x blur@
    - abs x n@ - abs < x blur@ - n@ + x ? ? - n@ +
""")

# okaychamp, vinverse2 optimized by me!
# https://discord.com/channels/856381934052704266/856383302097043497/946533873003679805
fuck_m2ts = core.akarin.Expr([edgefix, rg11], f"""
    {neutral} n! 4 255 * a! y y y y 2 * y[0,-1] y[0,1] + + 4 / - 2.7 * + y - n@ + sdiff! x
    y - n@ + diff! sdiff@ n@ - diff@ n@ - * 0 < sdiff@ n@ - abs diff@ n@ - abs < sdiff@ diff@ ?
    n@ - 0.65 * n@ + sdiff@ n@ - abs diff@ n@ - abs < sdiff@ diff@ ? ? n@ - + merge! x a@ +
    merge@ < x a@ + x a@ - merge@ > x a@ - merge@ ? ?
""")

denoise_bm3d = BM3DCuda(fuck_m2ts, [1.55, 0], 0, Profile.FAST, rg11, 2).clip
denoise_bm3d_hard = BM3DCuda(fuck_m2ts, [2.6, 0], 0, Profile.NORMAL, denoise_bm3d, 2).clip

denoise_bm3d = lvf.rfs(denoise_bm3d, denoise_bm3d_hard, hard_denoising_ranges)

to_descale = pad_reflect(denoise_bm3d, 22, 22, 22, 22)

with YUVPlanes(to_descale) as c:
    ys = depth(c.Y, 32)

    catr_desc720, what_desc720, catr_desc810, what_desc810 = [
        catrom.descale(ys, 1280, 720), pompobic.descale(ys, 1280, 720),
        catrom.descale(ys, 1440, 810), pompobic.descale(ys, 1440, 810)
    ]

    def _mix_descales(catr: vs.VideoNode, what: vs.VideoNode, op: ExprOp) -> vs.VideoNode:
        weighted = what.std.Convolution([1, 2, 1, 2, 4, 2, 1, 2, 1]).std.Convolution([1] * 9)
        median = what.ctmf.CTMF(2)

        return core.akarin.Expr([
            weighted, median, catr.std.MakeDiff(what), what, core.akarin.Expr(
                [what, weighted, median], 'x y - yd! x z - zd! yd@ zd@ * 0 < x yd@ abs zd@ abs < y z ? ?'
            )  # I'm sorry
        ], f"""
            b {ExprOp.convolution('b', [1, 2, 1, 2, 4, 2, 1, 2, 1])} - dblur! {ExprOp.matrix('z', False)}
            sort8 dup5 max_val! dup2 min_val! drop8 z dblur@ min_val@ min dblur@ max_val@ max clamp l0!
            l0@ abs z abs < l0@ z ? a + l! z d! x y - 2.49 * w! w@ 0.2 * w0! w@ 0.5 * w1! w0@ d@ * 0 < 0
            w0@ abs d@ abs < w0@ d@ ? ? l@ + l@ w1@ d@ * 0 < 0 w1@ abs d@ abs < w1@ d@ ? ? l@ + {op} {op}
        """)

    what_720_min = _mix_descales(catr_desc720, what_desc720, ExprOp.MIN)
    what_810_max = _mix_descales(catr_desc810, what_desc810, ExprOp.MAX)

    descale = core.akarin.Expr(
        [what_720_min, pompobic.scale(what_810_max, 1280, 720), pompobic.descale(what_810_max, 1280, 720)],
        '1.5 219 / t! y x min y! z x min z! x y@ - x z@ - xor x x y@ - abs x z@ - abs < z@ y@ t@ - y@ t@ + clamp z@ ? ?'
    )

    descale = depth(descale, 16, dither_type=Dither.ERROR_DIFFUSION)

    fsrcnnx = get_y(core.placebo.Shader(
        join([descale] * 3), stg.misc.x56_SHADERS, descale.width * 2, descale.height * 2, filter='box'
    ))

    smooth = Nnedi3SS(False, 0).scale(descale, descale.width * 2, descale.height * 2)

    halo_mask = lvf.mask.halo_mask(smooth)

    upscale = core.akarin.Expr(
        [fsrcnnx, smooth, core.std.Maximum(smooth), core.std.Minimum(smooth)], f"""
        x 0.8 * y 0.2 * + up! {1.5 * 2**8} O! {1.1 * 2**8} U! up@ z O@ + > z O@
        + up@ ? a U@ - < a U@ - up@ z O@ + > z O@ + up@ ? ? 16 255 * 235 255 * clamp
    """).std.MaskedMerge(
        core.akarin.Expr([smooth, fsrcnnx], 'x y min'), halo_mask
    )

    upscale = fine_dehalo(upscale, None, 2.1, 2.1, 0)

    pad_warp = upscale.warp.AWarpSharp2(None, 1.5, 1, 4)

    pad_rep = repair(pad_warp, upscale, 15)

    mask = PrewittStd().edgemask(upscale).akarin.Expr(
        f'x 4 255 * < 0 x 32 255 * > {peak} x ? ?'
    ).std.InvertMask().std.Convolution([1] * 9)

    final = repair(upscale.std.MaskedMerge(pad_rep, mask), upscale, 2)

    diff = upscale.std.MakeDiff(removegrain(upscale, 15))

    mask = PrewittStd().edgemask(diff.std.Levels(40 << 8, 168 << 8, 0.35))

    mask = mask.akarin.Expr(f"""
        x[-1,-1] x[1,1] min mi1! x[-1,-1] x[1,1] max ma1! x[0,-1] x[0,1] min mi2! x[0,-1] x[0,1] max ma2! x[1,-1]
        x[-1,1] min mi3! x[1,-1] x[-1,1] max ma3! x[-1,0] x[1,0] min mi4! x[-1,0] x[1,0] max ma4!
        x mi1@ ma1@ clamp clamp1! x mi2@ ma2@ clamp clamp2! x mi3@ ma3@ clamp clamp3! x mi4@ ma4@ clamp clamp4!
        x clamp1@ - abs ma1@ mi1@ - + c1! x clamp2@ - abs ma2@ mi2@ - + c2! x clamp3@ - abs ma3@ mi3@ - +
        c3! x clamp4@ - abs ma4@ mi4@ - + c4! c1@ c2@ c3@ c4@ min min min mindiff!
        mindiff@ c4@ = clamp4@ mindiff@ c2@ = clamp2@ mindiff@ c3@ = clamp3@ clamp1@ ? ? ? m!
        m@ 4 255 * < 0 m@ 16 255 * > {peak} m@ ? ?
    """)

    repaired = final.std.MaskedMerge(upscale, mask)

    singlerate = Eedi3SR(
        True, False, 0.15, 0.55, 500, nrad=3, mdis=10,
        eedi3_args=dict(vcheck=3), nnedi3_args=dict(nsize=0)
    ).aa(repaired.std.Transpose()).std.Transpose()

    singlerate = lvf.rfs(singlerate, repaired, no_aa_ranges)

    c.Y = catrom.scale(singlerate, ys.width, ys.height)

    sraa = upscaled_sraa(c.Y.std.Transpose(), 1.65, singlerater=Eedi3SR(
        True, False, 0.2, 0.6, 380, 0, 1,
        eedi3_args=dict(vcheck=0), nnedi3_args=dict(nsize=0)
    )).std.Transpose()

    c.Y = lvf.rfs(sraa, c.Y, no_aa_ranges)

    c.Y = fine_dehalo(c.Y, None, 2.2, 2.2, 0, 1.2, 40, 80)

rescaled = c.clip.std.Crop(22, 22, 22, 22)

rescaled = lvf.rfs(rescaled, denoise_bm3d, [*only_dehalo_1080p_ranges, *no_descale_ranges])

rescale_21_9_cropped = rescaled.std.Crop(0, 0, 104, 104).std.AddBorders(0, 0, 106, 102)

rescaled = lvf.rfs(rescaled, rescale_21_9_cropped, aspect_ratio_21_9_ranges)


def autistic_fixing(clip: vs.VideoNode) -> vs.VideoNode:
    aspect_ratio = 1036 / 1920 / 2

    down_720_1510 = Lanczos(5).scale(clip[1510], 1280, aspect_ratio * 1280 * 2)

    down_720_1510 = catrom.scale(down_720_1510, 720, aspect_ratio * 720 * 2)

    edi_fix_1510 = Eedi3SR(True, False, 0.4, 0.45, 340).aa(down_720_1510)

    upscale_1510 = Nnedi3SS(False, 0).scale(edi_fix_1510, 1920, 1036)

    fixing = insert_clip(clip, upscale_1510, 1510)

    # rescale ruined the text papers on the table
    fixing = replace_squaremask(fixing, denoise_bm3d, (479, 139, 1311, 897), (25922, 26020), 2.5)

    fixing = lvf.rfs(fixing, core.std.StackVertical([
        fixing.std.CropAbs(1920, 900),
        core.std.StackHorizontal([
            fixing.std.CropAbs(490, 136, y=900)
        ] * 4).std.CropAbs(1920, 136)
    ]), (26261, 26320))

    fixing = fixing.std.FreezeFrames(79057, 79057, 79058)
    fixing = fixing.std.FreezeFrames(85517, 85520, 85521)
    fixing = fixing.std.FreezeFrames(93237, 93284, 93237)

    clip_116049 = clip[116049:116096+1]

    manual_116049 = stg.src(r".\assets\manual_redraw_116049.png").resize.Bicubic(format=vs.YUV444P16, matrix=1)
    manual_ed_116049 = core.std.Expr(
        [clip_116049, manual_116049], ['x y max', 'y 1000 +']
    ).std.CropRel(bottom=102).bilateral.Gaussian(0.55).std.AddBorders(bottom=102)

    fix_116049 = replace_squaremask(
        clip_116049, manual_ed_116049, (252, 18, 833, 924), (0, None)
    )
    fix_116049 = fix_116049.std.CropRel(bottom=102).std.AddBorders(bottom=102)
    fix_116049 = core.std.Expr([fix_116049, manual_ed_116049, manual_116049], ['x', 'z y min'])

    fixing = insert_clip(fixing, fix_116049, 116049)

    return fixing


def manual_mask_1080p_elements(clip: vs.VideoNode) -> vs.VideoNode:
    mranges = onscreen_text_1080p_ranges

    binarized = lvf.rfs(
        edgefix.std.Binarize(57000),
        edgefix.std.Binarize(59900),
        mranges[9]
    )

    mask = core.std.Expr(binarized.std.SplitPlanes(), 'x y 2 * z 2 * max max')
    mask = mask.std.Maximum().std.Minimum().std.Maximum().std.Minimum().std.Maximum()
    mask = mask.std.Maximum().std.Minimum().std.Inflate().std.Maximum().std.Minimum()
    mask = mask.std.Maximum().std.Inflate().std.Maximum().std.Deflate().bilateral.Gaussian(1.55)
    mask = mask.std.Maximum()

    final_mask = mask.std.BlankClip()
    white_mask = final_mask.std.Invert()

    final_mask = lvf.rfs(final_mask, mask, mranges[0])
    final_mask = freeze_replace_mask(final_mask, mask.std.Inflate(), (685, 239, 80, 85), 1947, mranges[1])
    final_mask = replace_squaremask(final_mask, white_mask, (105, 105, 230, 100), mranges[1], 5)
    final_mask = freeze_replace_mask(final_mask, mask, (1090, 576, 830, 460), 1990, mranges[2])
    final_mask = final_mask[:mranges[3][0]] + mask[mranges[3][0]] * (mranges[3][1]+1 - mranges[3][0]) + (
        final_mask[mranges[3][1]+1:])
    final_mask = freeze_replace_mask(final_mask, mask, (629, 167, 1137, 77), 2467, mranges[4])
    final_mask = freeze_replace_mask(final_mask, mask, (416, 169, 205, 488), 2554, mranges[5])
    final_mask = freeze_replace_mask(final_mask, mask, (683, 217, 716, 50), 2722, mranges[6])
    final_mask = freeze_replace_mask(final_mask, mask, (415, 61, 760, 605), 2815, mranges[7])
    final_mask = lvf.rfs(final_mask, mask, mranges[8])
    final_mask = freeze_replace_mask(final_mask, mask, (987, 226, 114, 410), 3248, mranges[9])
    final_mask = lvf.rfs(final_mask, mask, mranges[10])

    return clip.std.MaskedMerge(denoise_bm3d, final_mask)


rescaled_444 = autistic_fixing(rescaled)
rescaled_444 = manual_mask_1080p_elements(rescaled_444)

rescaled = didee.resample(rescaled_444, vs.YUV420P16)

ccden = ccd(rescaled, 30)

with YUVPlanes(rescaled) as c:
    up, tr, th, st = 190 << 8,  4 << 8, 0.5 / 16, 48 / 128
    ref_clip = c.Y.akarin.Expr(f"""
        {peak} x {ExprOp.convolution('x', [16.0, 4.0, 16.0, 4.0, 1.0, 4.0, 16.0, 4.0, 16.0])}
        - x + - abs 2 sin * {neutral} -
    """)

    ref_clip = robisoft.scale(ref_clip, c.U.width, c.U.height)

    with YUVPlanes(ccden) as cc:
        def _deh(clip: vs.VideoNode, denclip: vs.VideoNode) -> vs.VideoNode:
            knlkwargs = dict(rclip=contrasharpening(clip, ref_clip), wmode=3)

            knlden = knl_means_cl(clip, 1.0, 0, **knlkwargs)

            denoise = BM3DCuda(knlden, 6, 0, Profile.NORMAL).clip

            comb = combine([knlden, denoise, denclip], ExprOp.MIN)

            rep = repair(repair(comb, knlden, 17), clip, 17)

            med = core.average.Mean([comb, rep, core.median.Median([knlden, denoise, denclip, comb, rep])])

            dehalo_bright = fine_dehalo(knlden, med, 2.2, 2.2, 0, 1.2, 40, 100)
            dehalo_dark = fine_dehalo(knlden, med, 2.2, 2.2, 0.12, 1.5, 0, 80)

            return core.akarin.Expr([
                knl_means_cl(dehalo_bright, 2.45, 0, 4, 2, **knlkwargs),
                knl_means_cl(dehalo_dark, 0.5, 0, 2, 1, **knlkwargs)
            ], """
                x y + 2 / val! val@ 60 255 * < val@ 40 255 * > val@ 60 - 20 255
                * / x * 1 val@ 60 - 20 255 * / - y * + x ? y ?
            """)

        c.U = _deh(c.U, cc.U)
        c.V = _deh(c.V, cc.V)

cdehalo = ccd(c.clip, 4)

# Custom stuffffff #

stitch_41580 = core.std.StackVertical([
    rescaled_444[41580], rescaled_444[41591]
])

pan_41580 = panner(
    cdehalo[41580:41592], stitch_41580,
    PanFunction(function_y=SineEaseIn)
)

cdehalo = insert_clip(cdehalo, pan_41580, 41580)

cropped_41641 = rescaled_444[41642].std.CropRel(right=177)
cropped_41641 = cropped_41641.std.AddBorders(left=176, right=1)

shifted_41641 = catrom.shift(cropped_41641, (0, 1.5))

stitch_41641 = core.std.StackHorizontal([
    rescaled_444[41641].std.CropAbs(250, rescaled_444.height),
    shifted_41641.std.CropRel(left=250, right=1920-880),
    rescaled_444[41641].std.CropRel(left=880),
    rescaled_444[41652]
])

pan_41641 = panner(
    cdehalo[41641:41653], stitch_41641,
    PanFunction(function_x=SineEaseIn)
)[1:]

pan_41641 = catrom.resample(pan_41641, pan_41641.format.id)

cdehalo = insert_clip(cdehalo, pan_41641, 41642)

stitch_70515 = core.std.StackHorizontal([
    rescaled_444[70515], rescaled_444[70527]
])

pan_70515 = panner(
    cdehalo[70515:70527], stitch_70515,
    PanFunction(function_x=SineEaseIn)
)

cdehalo = insert_clip(cdehalo, pan_70515, 70515)

stitch_70616 = core.std.StackHorizontal([
    rescaled_444[70616], rescaled_444[70627]
])

pan_70616 = panner(
    cdehalo[70616:70627], stitch_70616,
    PanFunction(function_x=SineEaseIn)
)

cdehalo = insert_clip(cdehalo, pan_70616, 70616)

clip_128177 = cdehalo[128177:128249+1]
clip444_128177 = rescaled_444[128177:128249+1]

stitch_128177 = core.std.StackVertical([
    clip444_128177[-1].std.CropAbs(1920, 720),
    clip444_128177[20].std.CropRel(0, 0, 23, 358).resize.Bicubic(src_top=0.5).std.CropRel(bottom=2),
    clip444_128177[0].std.CropRel(0, 0, 3),
])


pan_128177 = panner(
    clip_128177[:49], stitch_128177, PanFunction(
        PanDirection.INVERTED, function_y=QuadEaseInOut
    )
).std.DuplicateFrames([48]*(73-49))

cdehalo = insert_clip(cdehalo, pan_128177, 128177)

####################

deblock = lvf.deblock.vsdpir(pad_reflect(cdehalo, 0, 0, 2, 2), 15, tiles=4).std.Crop(0, 0, 2, 2)

cdehalo = lvf.rfs(cdehalo, deblock, deblocking_ranges)

with YUVPlanes(cdehalo) as c:
    deb_args = dict(
        downsample_h=720, topk=0.001, tvi_threshold=0.001,
        debug=(is_debug, False), ref_clip=get_y(fuck_m2ts), cambi_scale=3
    )

    adptvgr_args = dict(
        luma_scaling=8.5, static=False, size=0.95, sharp=45
    )

    y = c.Y

    light_deband = auto_deband(
        y, cambi_thr=1, min_thr=18, max_thr=32,
        steps=4, grain_thrs=(0.1, 0.13, 0.16),
        debander=f3kpf, debander_args=dict(
            limflt_args=dict(elast=4), radius=12
        ), adptvgr_args=adptvgr_args, **deb_args
    )

    hard_deband = auto_deband(
        light_deband, min_thr=48, max_thr=64,
        steps=6, grain_thrs=(0.18, 0.26, 0.2),
        debander=f3kbilateral, debander_args=dict(
            limflt_args=dict(thr=0.3), radius=15
        ), adptvgr_args=adptvgr_args, **deb_args
    )

    stronger_deband = auto_deband(
        hard_deband, min_thr=8, max_thr=12,
        steps=4, grain_thrs=(0.2, 0.3, 0.14),
        debander=placebo_deband, debander_args=dict(
            iterations=8, radius=16
        ), adptvgr_args=adptvgr_args, **deb_args
    )

    nuking_deband = f3kbilateral(
        stronger_deband, threshold=35,
        limflt_args=dict(thr=1, elast=2.65)
    )

    nuke2_lmfao = placebo_deband(
        nuking_deband, 16, 8, 10, 8
    )

    c.Y = lvf.rfs(hard_deband, stronger_deband, banding_galore_ranges)
    c.Y = lvf.rfs(c.Y, nuking_deband, twenty_tons_nuke_ranges)
    c.Y = lvf.rfs(c.Y, light_deband, light_debanding_ranges)
    c.Y = lvf.rfs(c.Y, nuke2_lmfao, idk_chief_deband_ranges)

deband = c.clip

normal_grain = Graigasm(
    thrs=[x << 8 for x in (40, 90, 168)],
    # Lower on dark because it's already being handled by the debanding
    strengths=[(0.15, 0), (0.45, 0), (0.3, 0)],
    sizes=(1.16, 0.95, 0.8),
    sharps=(55, 45, 50),
    grainers=[
        AddGrain(seed=80085, constant=False),
        AddGrain(seed=69420, constant=False),
        AddGrain(seed=69420, constant=False)
    ]
).graining(deband)

super_grain_y = Graigasm([235 << 8], [(0.25, 0)], [1.15], [100]).graining(crop)
super_grain_uv = Graigasm([240 << 8], [(0, 0.2)], [3], [65]).graining(crop)

super_grain = merge_chroma(super_grain_y, super_grain_uv)
super_grain = didee.resample(super_grain, vs.YUV420P16)


grain = lvf.rfs(normal_grain, super_grain, untouched_1080p_scenes_ranges)

decsized = decsiz(grain, 7, 0.009, 180 << 8, 240 << 8, 1.0, BilateralMethod.BILATERAL_GPU)
decsized = lvf.rfs(decsized, catrom.resample(denoise_bm3d, decsized.format), pepega_movie_end_range)
decsized = finalise_clip(decsized)
decsized = catrom.shift(
    catrom.resample(
        JPBD.clip[24:660], decsized.format
    ), (1, 0)
).std.Crop(0, 0, 20, 24) + decsized

if not is_debug:
    encoder = X265('x265_settings')
    encoder.resumable = True
    encoder.run_enc(decsized, JPBD, qpfile_clip=JPBD.clip[24:-42])
else:
    src_crop = catrom.shift(initialise_clip(JPBD.clip)[24:-42], (1, 0)).std.Crop(0, 0, 20, 24)

    stg.output(src_crop, False)
    stg.output(decsized, False)
