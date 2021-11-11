import stgfunc as stg
import lvsfunc as lvf
from vsdpir import DPIR
import vapoursynth as vs
from stgfunc import depth
from stgfunc.utils import replace_squaremask
from vardautomation import FileInfo, PresetAAC, PresetWEB
from selection_project_commons import SeleProFiltering, Encoding

core = vs.core

FUNI = FileInfo(r".\Source\[SubsPlease] Selection Project - 04 (1080p) [E77A2366].mkv", (240, 0), preset=(PresetWEB, PresetAAC))
FUNI.do_qpfile = True
BILI = FileInfo(r".\Source\[NC-Raws] SELECTION PROJECT - 04 [B-Global][WEB-DL][1080p][AVC AAC][Multiple Subtitle][MKV].mkv", preset=(PresetWEB))

OPSTART, OPENDIN = 2878, 5034


class Filtering(SeleProFiltering):
  def mix_sources(self):
    # MUSE Asia signs
    self.BILI.clip_cut = replace_squaremask(self.BILI.clip_cut, self.FUNI.clip_cut, (360, 150, 0, 930), [(4928, 5034), (32695, 32822)])

    # Bruuhhh
    self.BILI.clip_cut = lvf.rfs(self.BILI.clip_cut, self.FUNI.clip_cut, [
        (OPSTART + 378, OPSTART + 494), (OPSTART + 1197, OPSTART + 1208),
        (28823, 29003)
    ])

    # Funi shitting its pants as always
    self.FUNI.clip_cut = lvf.rfs(self.FUNI.clip_cut, self.BILI.clip_cut, [
        OPSTART + 344, (OPSTART + 346, OPSTART + 364),
        (OPSTART + 995, OPSTART + 1038), (OPSTART + 1183, OPSTART + 1186),
        OPSTART + 1196, OPSTART + 1159, OPSTART + 1303,
        (None, 431), (5179, 5180), (26368, 28822)
    ])

    # stg.output(lvf.diff(self.FUNI.clip_cut, self.BILI.clip_cut))

  def custom_scenefiltering(self, denoise, merge):
    rgb_32 = depth(merge.resize.Spline64(format=vs.RGB24, matrix_in=1), 32)
    rgb_dpir = DPIR(DPIR(rgb_32, 1.15), 20, 'deblock')
    dpir = rgb_dpir.resize.Spline64(format=denoise.format.id, matrix=1)

    dpir = dpir.grain.Add(0.46, 0.04, 0.065, 0.065, 69420, False)

    return lvf.rfs(denoise, dpir, (26299, 28874))

# SLOW: 26200


chain = Filtering(FUNI, BILI, (None, (30904, 32822)), True)
filtered = chain.filterchain()

if __name__ == '__main__':
  brrrr = Encoding(FUNI, filtered)
  brrrr.run()
else:
  stg.output(FUNI.clip_cut)
  stg.output(BILI.clip_cut)
  stg.output(filtered)
