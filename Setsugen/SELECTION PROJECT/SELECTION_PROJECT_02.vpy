import stgfunc as stg
import lvsfunc as lvf
import vapoursynth as vs
from vsutil import get_y
from typing import Dict, Any
from vsutil import insert_clip
from vardefunc.misc import merge_chroma
from stgfunc.utils import replace_squaremask
from vardautomation import FileInfo, PresetAAC, PresetWEB
from selection_project_commons import SeleProFiltering, Encoding

core = vs.core

FUNI = FileInfo(r".\Source\[SubsPlease] Selection Project - 02 (1080p) [C5DD71E5].mkv", (240, 0), preset=(PresetWEB, PresetAAC))
BILI = FileInfo(r".\Source\[NC-Raws] SELECTION PROJECT - 02 [B-Global][WEB-DL][1080p][AVC AAC][Multiple Subtitle][MKV].mkv", preset=(PresetWEB))


class Filtering(SeleProFiltering):
  def mix_sources(self):
    # MUSE Asia signs
    self.BILI.clip_cut = replace_squaremask(self.BILI.clip_cut, self.FUNI.clip_cut, (360, 150, 0, 930), [(0, 86), (33932, 34046)])

    # Funi shitting its pants as always
    self.FUNI.clip_cut = lvf.rfs(self.FUNI.clip_cut, self.BILI.clip_cut, (32884, 32927))

    for file in (self.FUNI, self.BILI):
      OP_AV1 = stg.src(r".\Extra\NCOP\SELECTION PROJECT OPテーマ 「Glorious Days」_AV1.mp4", ref=file.clip_cut)[:2158]
      OP_START, OP_ENDIN = (31889, 34046)

      texture = OP_AV1.grain.Add(20, 3, 0.07, 0.12, 69420, True).bilateral.Gaussian(1)

      merge = replace_squaremask(file.clip_cut[OP_START:OP_ENDIN], texture, (993, 50, 464, 516), (2042, None))
      merge = replace_squaremask(merge, OP_AV1, (624, 50, 833, 516), (2042, None))

      white = merge.std.BlankClip(length=1).std.Invert()

      merge = merge[:195] + white + merge[196:2041] + white + merge[2042:]

      file.clip_cut = insert_clip(
          file.clip_cut, merge + merge[-1], OP_START
      )

    # stg.output(lvf.diff(self.FUNI.clip_cut, self.BILI.clip_cut))

  def custom_scenefiltering(self, denoise: vs.VideoNode, merge: vs.VideoNode):
    sraa = lvf.sraa(denoise, 1.45)
    aa = replace_squaremask(denoise, sraa, (880, 1080, 880, 0), (9904, 9963))
    aa = replace_squaremask(aa, sraa, (55, 100, 1273, 541), (13304, 13434))
    aa = replace_squaremask(aa, sraa, (530, 352, 0, 728), (26742, 26807))
    aa = replace_squaremask(aa, sraa, (344, 562, 530, 518), (26742, 26807))

    def _eedi3_singlerate(clip: vs.VideoNode) -> vs.VideoNode:
      eeargs: Dict[str, Any] = dict(field=0, dh=False, alpha=1, beta=0, gamma=0, nrad=3, mdis=20)
      nnargs: Dict[str, Any] = dict(field=0, dh=False, nsize=0, nns=4, qual=2)
      y = get_y(clip)
      return lvf.aa.eedi3(sclip=lvf.aa.nnedi3(**nnargs)(y), **eeargs)(y)

    pepega = merge_chroma(lvf.sraa(
        aa.resize.Bicubic(height=1080 / 3), 1.8, width=1920, height=1080,
        aafun=_eedi3_singlerate
    ), aa)

    return replace_squaremask(aa, pepega, (64, 86, 1075, 625), [
        (9904, 9963), (10469, 10533)
    ])


chain = Filtering(FUNI, BILI)
filtered = chain.filterchain()

if __name__ == '__main__':
  brrrr = Encoding(FUNI, filtered)
  brrrr.run()
else:
  stg.output(FUNI.clip_cut)
  stg.output(BILI.clip_cut)
  stg.output(filtered)
