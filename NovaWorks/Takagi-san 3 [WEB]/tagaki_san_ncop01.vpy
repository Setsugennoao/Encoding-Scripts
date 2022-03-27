import stgfunc as stg
import lvsfunc as lvf
import vapoursynth as vs
from vsutil import iterate, get_y
from vardefunc.sharp import z4usm
from stgfunc.mask import getCreditMask
from vardefunc.util import finalise_clip
from stgfunc.transitions import fade_freeze
from lvsfunc.kernels import Catrom, Mitchell
from stgfunc.utils import replace_squaremask, freeze_replace_mask
from vardautomation import X265, FileInfo, PresetWEB, get_vs_core

from takagi_san_commons.filtering import TakagiSanSanFiltering

core = get_vs_core(range(0, vs.core.num_threads, 2))
minimum, maximum = core.std.Minimum, core.std.Maximum
catrom = Catrom()

fileinfo = FileInfo(
    r".\Source\01\Karakai Jouzu no Takagi-san S3 - 01 (Amazon dAnime VBR 1080p).mkv", preset=[PresetWEB]
)

sources = [
    stg.src(r".\Source\01\Karakai Jouzu no Takagi-san S3 - 01 (Amazon dAnime CBR 1080p).mkv", 16, matrix_prop=1),
    stg.src(r".\Source\01\Karakai Jouzu no Takagi-san S3 - 01 (Amazon dAnime VBR 1080p).mkv", 16, matrix_prop=1),
    stg.src(r".\Source\01\Karakai Jouzu no Takagi-san S3 - 01 (Netflix VBR 1080p).mkv", 16, matrix_prop=1)[24:-24]
]

sources = [source[1894:4052 + 1] for source in sources]
sources = [*sources[:2], core.average.Mean(sources), sources[2]]

ncop_sources = [
    stg.src(r".\Source\NCOP\TVアニメ『からかい上手の高木さん３』ノンクレジットOP「まっすぐ」／大原ゆい子-v8gAfylTw-w.webm", 16, matrix_prop=1),
    stg.src(r".\Source\NCOP\TVアニメ『からかい上手の高木さん３』ノンクレジットOP「まっすぐ」／大原ゆい子-v8gAfylTw-w.mp4", 16, matrix_prop=1)
]


amazon_cbr = sources[1]
ncop_avg = core.average.Mean(ncop_sources)

black_gray = get_y(amazon_cbr).std.BlankClip()
white_gray = black_gray.std.Invert()


def get_op_credits(mask: vs.VideoNode) -> vs.VideoNode:
    credit_mask = black_gray[:mask.num_frames + 1]

    credit_mask = freeze_replace_mask(credit_mask, mask, (767, 411, 587, 149), 156, (88, 171))
    credit_mask = replace_squaremask(credit_mask, white_gray, (106, 30, 1204, 379), (88, 171), 5)

    credit_mask = freeze_replace_mask(
        credit_mask, iterate(mask, minimum, 3).std.Maximum(),
        (559, 313, 147, 133), 533, (414, 539)
    )

    credit_mask = freeze_replace_mask(credit_mask, mask, (455, 191, 1315, 67), 650, (552, 677))
    credit_mask = freeze_replace_mask(credit_mask, mask, (617, 275, 143, 733), 650, (552, 677))

    credit_mask = freeze_replace_mask(credit_mask, mask, (1679, 937, 107, 59), 740, (684, 769))

    credit_mask = lvf.rfs(credit_mask, mask[879] * credit_mask.num_frames, (796, 885))

    credit_mask = lvf.rfs(credit_mask, mask[991] * credit_mask.num_frames, (991, 1140))

    credit_mask = lvf.rfs(credit_mask, mask[1361] * credit_mask.num_frames, (1327, 1422))

    mask_exp = iterate(iterate(mask, maximum, 6).std.Binarize(16 << 8), minimum, 6).std.BoxBlur()
    mask_exp2 = iterate(iterate(mask_exp, maximum, 18).std.Binarize(16 << 8), minimum, 15)

    credit_mask = lvf.rfs(credit_mask, mask_exp[1501] * credit_mask.num_frames, (1501, 1598))

    mask_exp = replace_squaremask(mask_exp, white_gray, (90, 40, 510, 767), 1767)
    credit_mask = lvf.rfs(credit_mask, mask_exp[1767] * credit_mask.num_frames, (1648, 1776))

    mask_exp = replace_squaremask(mask_exp, white_gray, (89, 42, 339, 221), 1821)
    credit_mask = lvf.rfs(credit_mask, mask_exp[1821] * credit_mask.num_frames, (1784, 1849))

    credit_mask = lvf.rfs(credit_mask, mask[1856].std.Maximum() * credit_mask.num_frames, (1856, 1907))

    credit_mask = lvf.rfs(credit_mask, mask_exp2[1989] * credit_mask.num_frames, (1915, 2009))

    credit_mask = lvf.rfs(credit_mask, mask_exp2[2135] * credit_mask.num_frames, (2018, 2157))

    return credit_mask


def get_ncop(ncop: vs.VideoNode, ep_op: vs.VideoNode, credit_mask: vs.VideoNode) -> vs.VideoNode:
    white = finalise_clip(core.std.BlankClip(ncop).std.Invert(), 16, True)
    ep_op_original = ep_op[::]

    # Apple
    ep_op = replace_squaremask(ep_op, white, (1920, 400, 0, 680), (228, 232))
    ep_op = replace_squaremask(ep_op, white, (725, 1080, 0, 0), (233, 240))
    ep_op = replace_squaremask(ep_op, white, (670, 1080, 1250, 0), (233, 240))
    ep_op = replace_squaremask(ep_op, white, (1366, 192, 269, 453), (240, 245))

    ep_op = lvf.rfs(ep_op, white.std.SetFrameProp('_Matrix', 1), (246, 389))

    shifted_233 = ep_op[0] + catrom.shift(
        Catrom(src_width=1921.5, src_height=1081.5).shift(ep_op, (-27.25, -.25)), (.5, -.5)
    )
    shifted_241 = catrom.shift(catrom.shift(ep_op[241], (83.25, -.25)), (.5, 0))
    sharp_grain_ncop = z4usm(ncop.grain.Add(.26, .05, .1, .1, 420, True), 1, 75)
    sharp_ncop = z4usm(ncop)
    wall_texture = Mitchell().scale(
        z4usm(catrom.scale(
            ncop.grain.Add(12, .05, .15, .15, 420, True), 1066, 600
        ), 2, 55), 1920, 1080
    ).bilateralgpu.Bilateral(1.25)

    ep_op = lvf.rfs(ep_op, shifted_233, 233)
    ep_op = replace_squaremask(ep_op, shifted_241 * ep_op.num_frames, (52, 81, 980, 564), 240)
    ep_op = replace_squaremask(ep_op, ep_op_original, (46, 68, 976, 599), 240)

    apple_234 = stg.src(
        r".\Source\NCOP_manual_edits\apple_234.png", 16, ep_op, matrix_prop=1
    ).bilateral.Bilateral(ep_op, 0.15, 1.5)
    apple_235 = catrom.shift(apple_234, (-56.5, 0))

    apple_236 = stg.src(r".\Source\NCOP_manual_edits\apple_236.png", 16, ep_op, matrix_prop=1)
    apple_237 = catrom.shift(apple_236, (-75, 0))

    apple_239 = stg.src(r".\Source\NCOP_manual_edits\apple_238.png", 16, ep_op, matrix_prop=1)
    apple_238 = replace_squaremask(catrom.shift(apple_239, (80, 0)), white, (1920, 80, 0, 1000), 238)

    ep_op = lvf.rfs(ep_op, apple_234, 234)
    ep_op = lvf.rfs(ep_op, apple_235, 235)
    ep_op = lvf.rfs(ep_op, apple_236, 236)
    ep_op = lvf.rfs(ep_op, apple_237, 237)
    ep_op = lvf.rfs(ep_op, apple_239, 239)
    ep_op = lvf.rfs(ep_op, apple_238, 238)

    # Finish Apple
    # Start title card transition

    ep_op = replace_squaremask(
        ep_op, fade_freeze(ep_op, ep_op, False, 389, 402), (1479, 297, 230, 349), (389, 402), 1.3
    )

    static_402 = core.average.Mean([ep_op[f] for f in [*range(402, 413 + 1), *range(540, 551 + 1)]])

    ep_op = replace_squaremask(ep_op, static_402, (205, 77, 205, 316), (414, 539))
    ep_op = replace_squaremask(ep_op, static_402, (47, 47, 464, 288), (414, 539))
    ep_op = replace_squaremask(ep_op, static_402, (76, 20, 259, 297), (414, 539))

    # Finish title card transition

    sharp_ncop = replace_squaremask(sharp_ncop, sharp_grain_ncop, (504, 177, 168, 165), (414, 539))

    static_552 = core.average.Mean([ep_op[f] for f in range(678, 683 + 1)])

    sharp_ncop = replace_squaremask(sharp_ncop, static_552, (366, 111, 1352, 102), (552, 677))
    sharp_ncop = replace_squaremask(sharp_ncop, static_552, (183, 284, 110, 676), (552, 677))

    sharp_ncop = replace_squaremask(sharp_ncop, wall_texture, (400, 378, 124, 0), (724, 769))

    sharp_ncop = replace_squaremask(sharp_ncop, sharp_grain_ncop, (655, 430, 1265, 650), (684, 723))

    return ep_op.std.MaskedMerge(sharp_ncop, credit_mask_op)


credit_mask = iterate(getCreditMask(amazon_cbr, ncop_avg, 30 << 8, 1.8, False), minimum, 6)

credit_mask_op = get_op_credits(credit_mask)

grain = TakagiSanSanFiltering().filtering([get_ncop(ncop_avg, clip, credit_mask) for clip in sources])

if __name__ == '__main__':
    encoder = X265('settings/x265_settings')

    fileinfo.name_clip_output = fileinfo.workdir / 'takagi_san_NCOP01.265'
    encoder.run_enc(grain, fileinfo)
else:
    stg.output(sources[0])
    stg.output(grain)
