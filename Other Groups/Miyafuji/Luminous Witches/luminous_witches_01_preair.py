from functools import partial
from math import floor
from typing import List

import vapoursynth as vs
import stgfunc as stg
from debandshit import dumb3kdb, f3kbilateral, placebo_deband
from jvsfunc import ccd
from lvsfunc import based_aa, dpir, hardsub_mask
from vardefunc import initialise_clip
from vsdehalo import contrasharpening, edge_cleaner, fine_dehalo
from vsdenoise import BM3DCudaRTC, MVTools, Profile
from vsencode import EncodeRunner, FileInfo
from vsutil import depth
from vsmask.edge import ExLaplacian4

core = vs.core


screencap_src = stg.src(r".\Source\01_preair\[My Shining Light] Luminous Witches #1 RAW.mkv")
chn_hards_fileinfo = FileInfo(r".\Source\01_preair\[Nekomoe kissaten][Luminous Witches][01][Pre-air][1080p][CHS].mp4")

screencap_src = initialise_clip(screencap_src, 8)
chn_hards_src = initialise_clip(chn_hards_fileinfo.clip, 8)


screencap_cut = screencap_src[410:]
chn_hards_cut = chn_hards_src[77:]


screencap0 = screencap_cut
screencap2 = screencap0.vivtc.VDecimate(2, False, 1.25, 10, 16, 16)
screencap5 = screencap2.vivtc.VDecimate(5, False, 1.05, 10, 8, 16)

factor_chn = chn_hards_cut.fps_num / chn_hards_cut.fps_den

factor0 = factor_chn * (screencap0.fps_den / screencap0.fps_num)
factor2 = factor_chn * (screencap0.fps_den / screencap2.fps_num)
factor5 = factor_chn * (screencap0.fps_den / screencap5.fps_num)

blankclip = chn_hards_cut.std.BlankClip(keep=True)

decimated_clips = list[vs.VideoNode]()

offsets = [-2, -1, 0, 1, 2]

decimation_match = [
    (factor0, screencap0),
    (factor2, screencap2),
    (factor5, screencap5)
]

for factor, cap_clip in decimation_match:
    decimated_clips.extend([
        blankclip.std.FrameEval(
            partial(
                lambda n, f, c, o, m: c[min(max(floor(n / f + o), 0), m)],
                f=factor, c=cap_clip, o=offset, m=cap_clip.num_frames - 1
            )
        ) for offset in offsets
    ])

one_pix = blankclip.std.BlankClip(1, 1, vs.GRAYS, keep=True)

planestats_clips = [decimated.std.PlaneStats(chn_hards_cut) for decimated in decimated_clips]

onepix_props_clips = [one_pix.std.CopyFrameProps(planestats) for planestats in planestats_clips]

diff_clips = [onepix_props.akarin.Expr('x.PlaneStatsDiff') for onepix_props in onepix_props_clips]

indices = list(range(len(diff_clips)))

merges_cache = dict[str, vs.VideoNode]()


def _select(n: int, f: List[vs.VideoFrame]) -> vs.VideoNode:
    scores = [diff[0][0, 0] for diff in f]  # type: ignore

    best_sorted = sorted(indices, key=lambda i: scores[i])

    best_idx = [
        i for i in best_sorted if (scores[i] - scores[best_sorted[0]]) < 0.00001
    ]

    if len(best_idx) == 1:
        return decimated_clips[best_idx[0]]

    cache_key = '_'.join(map(str, sorted(best_idx)))

    if cache_key not in merges_cache:
        merges_cache[cache_key] = core.average.Mean([decimated_clips[i] for i in best_idx])

    return merges_cache[cache_key]


decimated = blankclip.std.FrameEval(_select, diff_clips)
decimated = decimated.std.AssumeFPS(chn_hards_src)
decimated = chn_hards_src[:77] + decimated
decimated = stg.replace_squaremask(
    decimated, chn_hards_src, (250, 125, 1670, 955), (12580, 12745)  # lmao the steam notif
)

hards_mask = hardsub_mask(chn_hards_src, decimated, 0.05, inflate=6)
dehardsubbed = chn_hards_src.std.MaskedMerge(decimated, hards_mask)

screencap = depth(decimated, 16)
kissaten = depth(dehardsubbed, 16)

cden = ccd(kissaten, 5, 2)

denoise = BM3DCudaRTC(cden, [1.35, 0], 0, Profile.LOW_COMPLEXITY).clip

detailmask = stg.detail_mask(cden)

denoise = denoise.std.MaskedMerge(cden, detailmask)
denoise = contrasharpening(denoise, cden)

edgeclean = edge_cleaner(denoise, hot=True, smode=True)

dehalo = fine_dehalo(edgeclean)

mv = MVTools(dehalo, 1, 3)
mv.analyze(None, 16)

degrain = mv.degrain(None, 450)

deblock = dpir(degrain, 20, zones=[
    ((822, 953), 45),
    ((954, 1904), 75),
    ((1767, 1874), 150)
])

aa = based_aa(deblock, stg.x56_SHADERS)

deband = core.median.Median([
    dumb3kdb(aa, 8, 24, grain=3),
    dumb3kdb(aa, 16, 30, grain=3),
    placebo_deband(aa, grain=0),
])
deband = core.average.Mean([deband, f3kbilateral(deband, 16, 48)])

deband = deband.std.MaskedMerge(
    aa, ExLaplacian4().edgemask(aa)
).std.MaskedMerge(
    aa, detailmask
)


grain = stg.adaptive_grain(deband, 0.185, grainer=stg.Grainer.AddNoise)

if __name__ == '__main__':
    runner = EncodeRunner(chn_hards_fileinfo, grain)
    runner.video('x265', 'x265_settings_preair', qp_clip=False)
    # runner.audio('passthrough')
    runner.mux('Setsugen no ao @ Miyafuji')
    # runner.run(False)
    runner.patch((12580, 12745))
else:
    stg.output(kissaten)
    stg.output(grain)
