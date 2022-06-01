import stgfunc as stg
import lvsfunc as lvf
import EoEfunc as eoe
import kagefunc as kgf
import havsfunc as haf
import vapoursynth as vs
from stgfunc.fades import fade
from stgfunc.mask import getCreditMask
from stgfunc.deband import masked_f3kdb
from vsutil import get_y, depth, iterate
from vardefunc.noise import Graigasm, AddGrain
from debandshit import dumb3kdb, placebo_deband
from stgfunc.utils import replace_squaremask, freeze_replace_mask
from vardautomation import FileInfo, SelfRunner, X265, RunnerConfig

core = vs.core


def get_ep02_credit_mask(mask: vs.VideoNode) -> vs.VideoNode:
  credit_mask = black_gray[:mask.num_frames + 1]

  credit_mask = freeze_replace_mask(credit_mask, mask, (774, 396, 1058, 646), 516, (495, 559))

  credit_mask = freeze_replace_mask(credit_mask, mask, (637, 727, 1185, 181), 568, (560, 641))

  credit_mask = freeze_replace_mask(credit_mask, mask, (785, 555, 95, 269), 662, (642, 708))

  credit_mask = freeze_replace_mask(credit_mask, mask, (681, 353, 1159, 669), 714, (709, 768))

  credit_mask = freeze_replace_mask(credit_mask, mask, (893, 431, 127, 623), 850, (821, 898))

  credit_mask = freeze_replace_mask(credit_mask, mask, (1125, 521, 651, 523), 907, (899, 994))

  credit_mask = freeze_replace_mask(credit_mask, mask, (775, 493, 131, 33), 1014, (995, 1062))
  credit_mask = freeze_replace_mask(credit_mask, mask, (839, 587, 1019, 479), 1014, (995, 1062))

  credit_mask = freeze_replace_mask(credit_mask, mask, (571, 389, 685, 653), 1214, (1189, 1291))

  credit_mask = freeze_replace_mask(credit_mask, mask, (1767, 666, 80, 389), 1314, (1292, 1372))

  credit_mask = freeze_replace_mask(credit_mask, mask, (833, 455, 1021, 589), 1393, (1373, 1475))
  credit_mask = freeze_replace_mask(credit_mask, white_gray, (22, 18, 1261, 937), 1393, (1373, 1475))

  credit_mask = freeze_replace_mask(credit_mask, mask, (933, 817, 79, 69), 1598, (1575, 1689))
  credit_mask = freeze_replace_mask(credit_mask, white_gray, (25, 38, 211, 110), 1598, (1575, 1689))
  credit_mask = freeze_replace_mask(credit_mask, white_gray, (18, 21, 600, 133), 1598, (1575, 1689))
  credit_mask = freeze_replace_mask(credit_mask, white_gray, (19, 21, 318, 258), 1598, (1575, 1689))

  credit_mask = freeze_replace_mask(credit_mask, mask, (1021, 977, 761, 77), 1916, (1879, 1923))
  credit_mask = freeze_replace_mask(credit_mask, white_gray, (60, 40, 980, 110), 1916, (1879, 1923))
  credit_mask = freeze_replace_mask(credit_mask, white_gray, (23, 19, 1256, 130), 1916, (1879, 1923))
  credit_mask = freeze_replace_mask(credit_mask, white_gray, (29, 17, 1208, 357), 1916, (1879, 1923))

  credit_mask = freeze_replace_mask(credit_mask, mask, (615, 305, 1205, 713), 1941, (1924, 2041))
  credit_mask = freeze_replace_mask(credit_mask, white_gray, (196, 50, 1550, 751), 1941, (1924, 2041))
  credit_mask = freeze_replace_mask(credit_mask, white_gray, (60, 48, 1417, 763), 1941, (1924, 2041))
  credit_mask = freeze_replace_mask(credit_mask, white_gray, (43, 38, 1507, 771), 1941, (1924, 2041))

  credit_mask = freeze_replace_mask(credit_mask, mask, (1047, 129, 439, 481), 2058, (2042, 2157))

  return credit_mask


def get_ep03_credit_mask(mask: vs.VideoNode) -> vs.VideoNode:
  credit_mask = black_gray[:mask.num_frames + 1]

  credit_mask = freeze_replace_mask(credit_mask, mask, (563, 195, 689, 831), 495, (495, 543))

  credit_mask = freeze_replace_mask(credit_mask, mask, (563, 195, 689, 831), 555, (544, 614))

  credit_mask = freeze_replace_mask(credit_mask, mask, (599, 256, 81, 412), 666, (615, 676))

  credit_mask = freeze_replace_mask(credit_mask, mask, (709, 525, 1101, 299), 677, (677, 740))

  credit_mask = freeze_replace_mask(credit_mask, mask, (707, 137, 612, 863), 743, (741, 791))

  credit_mask = freeze_replace_mask(credit_mask, mask, (440, 245, 158, 185), 1024, (995, 1038))

  credit_mask = freeze_replace_mask(credit_mask, mask, (787, 407, 997, 639), 1071, (1039, 1086))

  credit_mask = freeze_replace_mask(credit_mask, mask, (571, 383, 685, 649), 1198, (1189, 1278))

  credit_mask = freeze_replace_mask(credit_mask, mask, (794, 553, 1053, 62), 1279, (1279, 1372))

  credit_mask = freeze_replace_mask(credit_mask, mask, (521, 499, 121, 319), 1575, (1575, 1689))
  credit_mask = freeze_replace_mask(credit_mask, white_gray, (24, 26, 435, 465), 1598, (1575, 1689))
  credit_mask = freeze_replace_mask(credit_mask, white_gray, (30, 27, 434, 485), 1598, (1575, 1689))

  credit_mask = freeze_replace_mask(credit_mask, mask, (1148, 200, 393, 824), 1770, (1690, 1770))

  credit_mask = freeze_replace_mask(credit_mask, mask, (595, 195, 669, 837), 1894, (1879, 1923))

  credit_mask = freeze_replace_mask(credit_mask, mask, (355, 87, 1367, 739), 1947, (1924, 2002))

  credit_mask = freeze_replace_mask(credit_mask, mask, (666, 60, 628, 961), 2069, (2042, 2157))

  return credit_mask


def src_merge(num_ep: int, crc: str, trim: slice) -> vs.VideoNode:
  funimation = stg.src(
      fr'..\..\Source\[SubsPlease] Selection Project - {num_ep:02d} (1080p) [{crc}].mkv', 16
  )[240:][trim]

  billibilli = stg.src(
      fr'..\..\Source\[NC-Raws] SELECTION PROJECT - {num_ep:02d} '
      fr'[B-Global][WEB-DL][1080p][AVC AAC][Multiple Subtitle][MKV].mkv', 16
  )[trim]

  muse_asia_logo = getCreditMask(funimation, billibilli, 25 << 8, None, True)

  billibilli = lvf.rfs(billibilli, billibilli.std.MaskedMerge(funimation, muse_asia_logo), (2043, None))

  return core.average.Mean([funimation, billibilli])


ep02 = src_merge(2, "C5DD71E5", slice(31889, None))

fileinfo = FileInfo(r"..\..\Source\[SubsPlease] Selection Project - 02 (1080p) [C5DD71E5].mkv")

last_frames = ep02[2155:]


def _add_last_frames(clip: vs.VideoNode) -> vs.VideoNode:
  if ep02.num_frames == clip.num_frames:
    return clip
  return clip + last_frames[-(ep02.num_frames - clip.num_frames):]


ep03 = _add_last_frames(src_merge(3, "4C3303CD", slice(480, 2638)))

OP_AV1 = stg.src(r".\Source\SELECTION PROJECT OPテーマ 「Glorious Days」_AV1.mp4", 16)[:2158]
OP_AVC = stg.src(r".\Source\Selection Project - Opening Theme - Glorious Days-Fjo2eR9MroY_AVC.mp4", 16)[:2158]

all_episodes = [ep02, ep03] + [
    _add_last_frames(x) for x in [
        src_merge(4, "E77A2366", slice(2878, 5034)),
        src_merge(5, "D029FCC7", slice(1272, 3428))[1:],
        src_merge(6, "7943CAF4", slice(2710, 4866)),
        src_merge(7, "E87C7ED7", slice(1990, 4148)),
        src_merge(8, "705C5EFC", slice(2590, 4746)),
        src_merge(9, "7EC82FDF", slice(1176, 3332)),
        src_merge(10, "FAD986BF", slice(1462, 3620))[1:],
        src_merge(11, "A50CC4E9", slice(624, 2781))
    ]
]

black_gray = get_y(ep02).std.BlankClip()
white_gray = black_gray.std.Invert()

black_yuv = ep02.std.BlankClip()
white_yuv = black_yuv.std.Invert()


def get_ncop() -> vs.VideoNode:
  mask_nced = kgf.kirsch(get_y(OP_AVC))

  ncop_logos = replace_squaremask(
      core.std.Expr([
          mask_nced[0], mask_nced[195], mask_nced[2040]
      ], 'x y + z +'), black_gray[0], (1920, 700, 0, 180), 0
  )

  clean_angles = lvf.rfs(ep02, ep03, [(995, 1062), (1575, 1689)])

  ncop_logos = iterate(ncop_logos, lambda x: x.std.Maximum().std.Minimum(), 20)

  ncop_logos = iterate(ncop_logos, core.std.Maximum, 3).std.Inflate().std.Binarize()

  ncop_logos = replace_squaremask(ncop_logos, white_gray, (30, 25, 1790, 60), 0)

  ncop_logos = ncop_logos * OP_AVC.num_frames

  ncop_nl = core.std.MaskedMerge(OP_AVC, clean_angles, ncop_logos)

  smdeg = haf.SMDegrain(ncop_nl, 1, 150, 50)

  return eoe.dn.BM3D(smdeg, 12)


ncop_nologo = get_ncop()


def get_textured() -> vs.VideoNode:
  textured = OP_AV1.grain.Add(20, 3, 0.07, 0.12, 69420, True)

  textured_masked = OP_AV1.std.MaskedMerge(
      textured.bilateral.Gaussian(1.25),
      get_y(OP_AV1).std.Binarize(175 << 8).std.Invert().std.Inflate()
  )

  def do_texture(f, n):
    if n >= 2066:
      return textured_masked
    elif 2042 <= n <= 2065:
      return textured.bilateral.Gaussian((2065 - 2042) / (n - 2041) * 2)

  return OP_AV1.std.FrameEval(do_texture, OP_AV1)


texture = get_textured()


def get_clean_start() -> vs.VideoNode:
  ep3_start = replace_squaremask(ep03, OP_AV1.bilateral.Gaussian(1.5), (727, 50, 599, 516), (None, 195), 1)

  cmask = getCreditMask(ep02, ep3_start, 40 << 8, 0.35, False)

  cmaskep2 = lvf.rfs(black_gray, cmask, [(60, None)])

  cmaskep2 = freeze_replace_mask(cmaskep2, cmask, (532, 168, 700, 461), 13, (0, 59))

  ep2_mask = replace_squaremask(cmaskep2, white_gray, (24, 24, 865, 570), (22, 67))
  ep2_start = ep02.std.MaskedMerge(ep3_start, ep2_mask)

  return core.average.Mean([
      ep2_start, ep3_start,
      dumb3kdb(ep2_start, 8, 18),
      dumb3kdb(ep3_start, 8, 18),
      dumb3kdb(ep2_start, 12, 24),
      dumb3kdb(ep3_start, 12, 24),
      dumb3kdb(ep2_start, 18, 30),
      dumb3kdb(ep3_start, 18, 30),
      dumb3kdb(ep2_start, 24, 34),
      dumb3kdb(ep3_start, 24, 34),
      placebo_deband(ep2_start, 8),
      placebo_deband(ep3_start, 8),
      placebo_deband(ep2_start, 12),
      placebo_deband(ep3_start, 12),
      placebo_deband(ep2_start, 16),
      placebo_deband(ep3_start, 16)
  ])


clean_start = get_clean_start()

for i, ep in enumerate(all_episodes):
  ep = lvf.rfs(ep, clean_start, (None, 195))
  ep = replace_squaremask(ep, texture, (993, 50, 464, 516), (2042, None), 0.95)
  ep = replace_squaremask(ep, OP_AV1, (624, 50, 833, 516), (2042, None), 1)
  ep = replace_squaremask(ep, ep02, (670, 60, 627, 962), (2042, None), 1)

  if i > 0:
    ep = replace_squaremask(ep, ep[2003] * 50, (311, 49, 1390, 756), (1987, 2002))

  all_episodes[i] = ep

credit_masks = [
    getCreditMask(clip, ncop_nologo, 29 << 8, 1.55) for clip in [ep02, ep03]
]

ep02_credit_mask = get_ep02_credit_mask(credit_masks[0]).std.BinarizeMask(24 << 8)
ep03_credit_mask = get_ep03_credit_mask(credit_masks[1]).std.BinarizeMask(24 << 8)

mix_ep02_mask = core.std.Expr([ep02_credit_mask, ep03_credit_mask], 'x y -')
mix_ep03_mask = core.std.Expr([ep02_credit_mask, ep03_credit_mask], 'y x -')

inverted_mask_ep02 = black_gray.std.MaskedMerge(ep03_credit_mask, ep02_credit_mask)
inverted_mask_ep03 = black_gray.std.MaskedMerge(ep02_credit_mask, ep03_credit_mask)


def merge_into_ep02(clip_to_merge: vs.VideoNode, mask: vs.VideoNode) -> vs.VideoNode:
  all_episodes[0] = all_episodes[0].std.MaskedMerge(clip_to_merge, mask)


def merge_into_eps(clip_to_merge: vs.VideoNode, mask: vs.VideoNode) -> vs.VideoNode:
  all_episodes[1:] = [clip.std.MaskedMerge(clip_to_merge, mask) for clip in all_episodes[1:]]


both_dehard_ranges = [(495, 641), (677, 708), (741, 820), (1279, 1372)]
overlap_ranges = [(642, 676), (709, 740), (995, 1062), (1575, 1689), (1879, 1986)]

merged_ep02_cmask = lvf.rfs(
    black_gray, ep02_credit_mask, both_dehard_ranges + [(821, 994), (1373, 1475), (1987, 2041)]
)
merged_ep02_cmask = lvf.rfs(merged_ep02_cmask, mix_ep02_mask, overlap_ranges)

merged_ep03_cmask = lvf.rfs(
    black_gray, ep03_credit_mask, both_dehard_ranges + [(1063, 1086), (1690, 1770), (2042, 2157)]
)
merged_ep03_cmask = lvf.rfs(merged_ep03_cmask, mix_ep03_mask, overlap_ranges)

merge_into_ep02(core.average.Mean(all_episodes[1:]), merged_ep02_cmask)
merge_into_eps(all_episodes[0], merged_ep03_cmask)

# Credits are the same
ncop_ep02_cmask = lvf.rfs(black_gray, ep03_credit_mask, (1189, 1278))
ncop_ep03_cmask = lvf.rfs(black_gray, ep03_credit_mask, (1189, 1278))

# Credits in both and they overlap, merge ncop from YT
ncop_ep02_cmask = lvf.rfs(ncop_ep02_cmask, inverted_mask_ep02, overlap_ranges)
ncop_ep03_cmask = lvf.rfs(ncop_ep03_cmask, inverted_mask_ep03, overlap_ranges)

merge_into_ep02(ncop_nologo, ncop_ep02_cmask)
merge_into_eps(ncop_nologo, ncop_ep03_cmask)

mean_average = core.average.Mean(all_episodes)

best_select = stg.other.bestframeselect(all_episodes, mean_average)

freqmerge = stg.oyster.Core().FreqMerge(best_select, mean_average)

halo_mask = lvf.mask.halo_mask(freqmerge, rad=1, brz=0.85, thmi=0.35, thma=0.95)
halo_mask = halo_mask.std.Maximum().std.Inflate()

fmerge_smd = haf.SMDegrain(freqmerge, 1)

ref_bil = freqmerge.bilateral.Bilateral(None, 10, 2 / 255)

bidh = freqmerge.bilateral.Bilateral(ref_bil, 10 / 3, 2 / 255)

dehalo2 = core.std.Expr([fmerge_smd, bidh], "x y min")

dehalo3 = lvf.dehalo.masked_dha(dehalo2, 1, 1, 0.86, 0, 86, 24)

dehalo_edgc = haf.EdgeCleaner(dehalo3, 6, True, 16, 1, True)

dehalo = core.std.MaskedMerge(freqmerge, dehalo_edgc, halo_mask)

deband = masked_f3kdb(dehalo, rad=16, thr=[30, 24])

pref = iterate(get_y(dehalo), core.std.Maximum, 2).std.Convolution([1] * 9)

seed = 69420

grain = Graigasm(
    thrs=[x << 8 for x in (56, 85, 150)],
    strengths=[(0.65, 0.10), (0.48, 0.08), (0.12, 0)],
    sizes=(1.14, 1.23, 1.19),
    sharps=(50, 55, 45),
    grainers=[
        AddGrain(seed=seed, constant=False),
        AddGrain(seed=seed, constant=False),
        AddGrain(seed=seed, constant=True)
    ]
).graining(deband, prefilter=pref)

heavy_grain_params = dict(
    thrs=[x << 8 for x in (50, 90, 135)],
    strengths=[(1.4, 0.2), (1.27, 0.12), (1.12, 0)],
    sizes=(3, 2.58, 2.1),
    sharps=(70, 85, 55)
)

heavy_grain = Graigasm(
    **heavy_grain_params,
    grainers=[
        AddGrain(seed=seed, constant=False),
        AddGrain(seed=seed, constant=False),
        AddGrain(seed=seed, constant=True)
    ]
).graining(deband, prefilter=pref)

heavy_static_grain = Graigasm(
    **heavy_grain_params,
    grainers=[
        AddGrain(seed=seed, constant=True),
        AddGrain(seed=seed, constant=True),
        AddGrain(seed=seed, constant=True)
    ]
).graining(deband, prefilter=pref)

grain = fade(grain, heavy_grain, True, 815, 839)
grain = lvf.rfs(grain, heavy_grain, [(840, 987)])
grain = fade(grain, heavy_grain, False, 988, 1000)

grain = fade(grain, heavy_static_grain, True, 1290, 1293)
grain = lvf.rfs(grain, heavy_static_grain, [(1294, 1300)])
grain = fade(grain, heavy_static_grain, False, 1301, 1304)

grain = depth(grain, 10).std.Limiter(16 << 2, [235 << 2, 240 << 2], [0, 1, 2])

if __name__ == "__main__":
  encoder = X265("x265_settings_NCOP")

  fileinfo.name_clip_output = fileinfo.workdir / 'SELEPRO_NCOP_v1.265'

  SelfRunner(grain, fileinfo, RunnerConfig(encoder)).run()
else:
  stg.output(ep02)
  stg.output(mean_average)
  stg.output(deband)
  stg.output(grain)


# Credits
# 642-676 Overlap
# 709-740 Overlap
# 995-1062 Overlap
# 1575-1689 Overlap
# 1879-1986 Overlap

# Episode 2
# 495-559 ✅
# 560-641 ✅
# 642-708 ✅
# 709-768 ✅
# 821-898 ✅
# 899-994 ✅
# 995-1062 ✅
# 1189-1291 ✅
# 1292-1372 ✅
# 1373-1475 ✅
# 1575-1689 ✅
# 1879-1923 ✅
# 1924-2041
#   1924-1986 ✅ (reds) | 1987-2004 ✅ | 2005-2041 ✅
# 2042-2157 ✅

# Episode 3~11
# 495-543 ✅
# 544-614 ✅
# 615-676
#   615-641 ✅ | 642-676 ✅
# 677-740
#   677-708 ✅ | 709-740 ✅
# 741-791 ✅
# 995-1038 ✅
# 1039-1086 ✅
# 1189-1278 ✅
# 1279-1372 ✅
# 1575-1689 ✅
# 1690-1770 ✅
# 1879-1923 ✅
# 1924-2002
#   1924-1986 ✅ (reds) | 1987-2002 ✅
# 2042-2157 ✅
#
# def merge_ep02_in_03(
#     frame_range: Tuple[int, int], mask: vs.VideoNode,
#     clip_to_merge: vs.VideoNode = all_episodes[0]
# ) -> vs.VideoNode:
#   all_episodes[1:] = [
#       lvf.rfs(clip, clip.std.MaskedMerge(clip_to_merge, mask), frame_range)
#       for clip in all_episodes[1:]
#   ]
#
#
# def merge_alls_in_02(
#     frame_range: Tuple[int, int], mask: vs.VideoNode,
#     clip_to_merge: vs.VideoNode = core.average.Mean(all_episodes[1:])
# ) -> vs.VideoNode:
#   all_episodes[0] = lvf.rfs(
#       all_episodes[0], all_episodes[0].std.MaskedMerge(
#           clip_to_merge, mask
#       ), frame_range
#   )
#
# Credits are the same
# for frame_range in [(1189, 1278)]:
#   merge_ep02_in_03(frame_range, ep03_credit_mask, ncop_nologo)
#   merge_alls_in_02(frame_range, ep03_credit_mask, ncop_nologo)

# # do_overlap = False -> Credits in different places
# # do_overlap = True  -> Credits in both and they overlap, also merge ncop from YT
# for frame_range, do_overlap in [
#     ((495, 641), False), ((642, 676), True),
#     ((677, 708), False), ((709, 740), True),
#     ((741, 820), False),
#     ((995, 1062), True), ((1279, 1372), False),
#     ((1575, 1689), True), ((1879, 1986), True)
# ]:
#   if do_overlap:
#     merge_ep02_in_03(frame_range, mix_ep03_mask)
#     merge_alls_in_02(frame_range, mix_ep02_mask)
#     merge_ep02_in_03(frame_range, inverted_mask_ep03, ncop_nologo)
#     merge_alls_in_02(frame_range, inverted_mask_ep02, ncop_nologo)
#   else:
#     merge_ep02_in_03(frame_range, ep03_credit_mask)
#     merge_alls_in_02(frame_range, ep02_credit_mask)

# # Credits only in 02
# for frame_range in [(821, 994), (1373, 1475), (1987, 2041)]:
#   merge_alls_in_02(frame_range, ep02_credit_mask)

# # Credits only in 03
# for frame_range in [(1063, 1086), (1690, 1770), (2042, 2157)]:
#   merge_ep02_in_03(frame_range, ep03_credit_mask)
