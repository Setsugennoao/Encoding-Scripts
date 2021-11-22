import stgfunc as stg
import lvsfunc as lvf
from vsdpir import DPIR
import vapoursynth as vs
from stgfunc import depth
from stgfunc.utils import replace_squaremask
from vardautomation import FileInfo, PresetAAC, PresetWEB
from selection_project_commons import SeleProFiltering, Encoding

core = vs.core

FUNI = FileInfo(r".\Source\[SubsPlease] Selection Project - 05 (1080p) [D029FCC7].mkv", (240, 0), preset=(PresetWEB, PresetAAC))
FUNI.do_qpfile = True
BILI = FileInfo(r".\Source\[NC-Raws] SELECTION PROJECT - 05 [B-Global][WEB-DL][1080p][AVC AAC][Multiple Subtitle][MKV].mkv", preset=(PresetWEB))

OPSTART, OPENDIN = 1272, 3428


class Filtering(SeleProFiltering):
  def mix_sources(self):
    # MUSE Asia signs
    self.BILI.clip_cut = replace_squaremask(self.BILI.clip_cut, self.FUNI.clip_cut, (360, 150, 0, 930), [(3318, 3428), (32696, 32823)])

    # Bruuhhh
    self.BILI.clip_cut = lvf.rfs(self.BILI.clip_cut, self.FUNI.clip_cut, [
        (OPSTART + 378, OPSTART + 494), (OPSTART + 1197, OPSTART + 1208),
        (27052, 27191), 27229, (27358, 27362), (2529, 2561)
    ])

    # Funi shitting its pants as always
    self.FUNI.clip_cut = lvf.rfs(self.FUNI.clip_cut, self.BILI.clip_cut, [
        OPSTART + 344, (OPSTART + 346, OPSTART + 364),
        (OPSTART + 995, OPSTART + 1038), (OPSTART + 1183, OPSTART + 1186),
        OPSTART + 1196, OPSTART + 1159, OPSTART + 1303, (2527, 2528),
        (27192, 27228), 27230, (27258, 27357), (27363, 29444), (2564, 2565)
    ])

    # stg.output(lvf.diff(self.FUNI.clip_cut, self.BILI.clip_cut))

  def custom_scenefiltering(self, denoise, merge):
    rgb_32 = depth(merge.resize.Spline64(format=vs.RGB24, matrix_in=1), 32)
    rgb_den_dpir = DPIR(rgb_32, 1.18)
    rgb_deb_dpir = DPIR(rgb_den_dpir, 22, 'deblock')
    den_dpir = rgb_den_dpir.resize.Spline64(format=denoise.format.id, matrix=1)
    deb_dpir = rgb_deb_dpir.resize.Spline64(format=denoise.format.id, matrix=1)

    dpir = deb_dpir.grain.Add(0.56, 0.045, 0.055, 0.055, 69420, False)

    noise = den_dpir.grain.Add(8.46, 13.84, 0.385, 0.185, 69420, False)
    noise_low = den_dpir.grain.Add(10.16, 2.08, 0.085, 0.085, 69420, False)

    rfs_noise = lvf.rfs(denoise, noise, (27223, 27230))
    rfs_noise = lvf.rfs(rfs_noise, noise_low, 27231)

    return lvf.rfs(rfs_noise, dpir, [(27136, 27222), (27232, 29444)])


chain = Filtering(FUNI, BILI, ((OPSTART, OPENDIN), (30905, 32823)), True)
filtered = chain.filterchain()

if __name__ == '__main__':
  brrrr = Encoding(FUNI, filtered)
  brrrr.run()
else:
  stg.output(FUNI.clip_cut)
  stg.output(BILI.clip_cut)
  stg.output(filtered)
