import stgfunc as stg
import lvsfunc as lvf
import vapoursynth as vs
from vsutil import get_y
from stgfunc.utils import replace_squaremask
from vardautomation import FileInfo, SelfRunner, x265, RunnerConfig

from selepro_ncs_commons.masking import getCreditMask

core = vs.core


def src_merge(num_ep: int, crc: str, trim: slice) -> vs.VideoNode:
  funimation = stg.src(
      fr'..\..\Source\[SubsPlease] Selection Project - {num_ep:02d} (1080p) [{crc}].mkv', 16
  )[240:][trim]

  billibilli = stg.src(
      fr'..\..\Source\[NC-Raws] SELECTION PROJECT - {num_ep:02d} '
      fr'[B-Global][WEB-DL][1080p][AVC AAC][Multiple Subtitle][MKV].mkv', 16
  )[trim]

  muse_asia_logo = getCreditMask(funimation, billibilli, 25, None, True)

  billibilli = lvf.rfs(billibilli, billibilli.std.MaskedMerge(funimation, muse_asia_logo), (1791, None))

  return core.average.Mean([funimation, billibilli])


fileinfo = FileInfo(r"..\..\Source\[SubsPlease] Selection Project - 03 (1080p) [4C3303CD].mkv")

all_episodes = [
    src_merge(3, "4C3303CD", slice(30929, 32847)),
    src_merge(4, "E77A2366", slice(30904, 32822)),
    src_merge(5, "D029FCC7", slice(30905, 32823)),
    src_merge(6, "7943CAF4", slice(31552, 33469)),
    src_merge(7, "E87C7ED7", slice(30115, 32033)),
    src_merge(8, "705C5EFC", slice(31216, 33132)),
    src_merge(9, "7EC82FDF", slice(30426, 32344)),
    src_merge(10, "FAD986BF", slice(31433, 33350)),
    src_merge(11, "A50CC4E9", slice(31722, 33639))
]

black_gray = get_y(all_episodes[0]).std.BlankClip()
white_gray = black_gray.std.Invert()

black_yuv = all_episodes[0].std.BlankClip()
white_yuv = black_yuv.std.Invert()

for i, ep in enumerate(all_episodes):
  all_episodes[i] = replace_squaremask(ep, white_yuv, (960, 1080, 0, 0), [
      (0, 242), (514, 770)
  ])


if __name__ == "__main__":
  grain = None
  encoder = x265("x265_settings_NCED")

  fileinfo.name_clip_output = fileinfo.workdir / 'SELEPRO_NCED_v1.265'

  SelfRunner(grain, fileinfo, RunnerConfig(encoder)).run()
else:
  for ep in all_episodes:
    stg.output(ep, False)

# Episode 03
# 0-242 ✅
# 243-314
# 315-386
# 387-458
# 514-770 ✅
# 867-938
# 939-1010
# 1011-1082
# 1083-1154
# 1155-1226
# 1227-1299
# 1300-1391
# 1392-1463
# 1464-1535
# 1536-1663
