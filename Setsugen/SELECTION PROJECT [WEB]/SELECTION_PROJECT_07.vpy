import stgfunc as stg
import lvsfunc as lvf
from vsdpir import DPIR
import vapoursynth as vs
from stgfunc import depth
from stgfunc.utils import replace_squaremask
from vardautomation import FileInfo, PresetAAC, PresetWEB
from selection_project_commons import SeleProFiltering, Encoding

core = vs.core

FUNI = FileInfo(r".\Source\[SubsPlease] Selection Project - 07 (1080p) [E87C7ED7].mkv", (240, 0), preset=(PresetWEB, PresetAAC))
BILI = FileInfo(r".\Source\[NC-Raws] SELECTION PROJECT - 07 [B-Global][WEB-DL][1080p][AVC AAC][Multiple Subtitle][MKV].mkv", preset=(PresetWEB))

OPSTART, OPENDIN = 1990, 4148


class Filtering(SeleProFiltering):
  def mix_sources(self):
    # MUSE Asia signs
    self.BILI.clip_cut = replace_squaremask(self.BILI.clip_cut, self.FUNI.clip_cut, (360, 150, 0, 930), [(4038, 4148), (31906, 32033)])

    # Bruuhhh
    self.BILI.clip_cut = lvf.rfs(self.BILI.clip_cut, self.FUNI.clip_cut, [
        (OPSTART + 378, OPSTART + 494), (OPSTART + 1197, OPSTART + 1208),
        (7657, 7751), (3244, 3268)
    ])

    # Funi shitting its pants as always
    self.FUNI.clip_cut = lvf.rfs(self.FUNI.clip_cut, self.BILI.clip_cut, [
        OPSTART + 344, (OPSTART + 346, OPSTART + 364),
        (OPSTART + 995, OPSTART + 1038), (OPSTART + 1183, OPSTART + 1186),
        OPSTART + 1196, OPSTART + 1159, OPSTART + 1303, (29476, 29495),
        (28474, 28930)
    ])

    # stg.output(lvf.diff(self.FUNI.clip_cut, self.BILI.clip_cut))

  def custom_scenefiltering(self, denoise, merge):
    # 26913-29187

    rgb_32 = depth(self.BILI.clip_cut.resize.Spline64(format=vs.RGB24, matrix_in=1), 32)

    rgb_den_dpir = lvf.rfs(DPIR(rgb_32, 1.16), rgb_32, [
        (28307, 28353), (28688, 28708)
    ])

    rgb_deb_dpir = lvf.rfs(DPIR(rgb_den_dpir, 20, 'deblock'), rgb_den_dpir, [
        (29126, 29187)
    ])

    deb_dpir = rgb_deb_dpir.resize.Spline64(format=denoise.format.id, matrix=1)

    return lvf.rfs(denoise, deb_dpir, [
        (26997, 27192), (27284, 27342), (27386, 27667), (27683, 27708),
        (27726, 27917), (28017, 28484), (28497, 29187)
    ])


chain = Filtering(FUNI, BILI, ((OPSTART, OPENDIN), (30115, 32033)), True)
filtered = chain.filterchain()

if __name__ == '__main__':
  brrrr = Encoding(FUNI, filtered)
  brrrr.run()
else:
  stg.output(FUNI.clip_cut)
  stg.output(BILI.clip_cut)
  stg.output(filtered)
