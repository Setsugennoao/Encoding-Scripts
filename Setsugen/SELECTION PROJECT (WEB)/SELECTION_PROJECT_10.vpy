import stgfunc as stg
import lvsfunc as lvf
from vsdpir import DPIR
from vsutil import depth
import vapoursynth as vs
from stgfunc.utils import replace_squaremask
from selection_project_commons import SeleProFiltering, Encoding
from vardautomation import FileInfo, PresetAAC, PresetWEB, get_vs_core

core = get_vs_core(range(0, vs.core.num_threads, 2))

FUNI = FileInfo(
  r".\Source\[SubsPlease] Selection Project - 10 (1080p) [FAD986BF].mkv", (240, 0), preset=(PresetWEB, PresetAAC)
)
BILI = FileInfo(
    r".\Source\[NC-Raws] SELECTION PROJECT - 10 [B-Global][WEB-DL][1080p][AVC AAC][Multiple Subtitle][MKV].mkv",
    preset=(PresetWEB)
)

OPSTART, OPENDIN = 1462, 3620


class Filtering(SeleProFiltering):
    def mix_sources(self):
        # MUSE Asia signs
        self.BILI.clip_cut = replace_squaremask(
          self.BILI.clip_cut, self.FUNI.clip_cut, (360, 150, 0, 930), [(3510, 3620), (33224, 33350)]
        )

        # Bruuhhh
        self.BILI.clip_cut = lvf.rfs(self.BILI.clip_cut, self.FUNI.clip_cut, [
            (OPSTART + 378, OPSTART + 494), (OPSTART + 1197, OPSTART + 1208)
        ])

        # Funi shitting its pants as always
        self.FUNI.clip_cut = lvf.rfs(self.FUNI.clip_cut, self.BILI.clip_cut, [
            OPSTART + 344, (OPSTART + 346, OPSTART + 364),
            (OPSTART + 995, OPSTART + 1038), (OPSTART + 1183, OPSTART + 1186),
            OPSTART + 1196, OPSTART + 1159, OPSTART + 1303
        ])

        # stg.output(lvf.diff(self.FUNI.clip_cut, self.BILI.clip_cut))

    def custom_scenefiltering(self, denoise, merge):
        rgb_32 = depth(merge.resize.Spline64(format=vs.RGB24, matrix_in=1), 32)
        rgb_deb_dpir = DPIR(DPIR(rgb_32, 1.18), 10, 'deblock')
        deb_dpir = rgb_deb_dpir.resize.Spline64(format=denoise.format.id, matrix=1)

        return lvf.rfs(denoise, deb_dpir, (8726, 8905))


chain = Filtering(FUNI, BILI, ((OPSTART, OPENDIN), (31433, 33350)), True)
filtered = chain.filterchain()

if __name__ == '__main__':
    brrrr = Encoding(FUNI, filtered[1809:-25324])  # , 1)
    brrrr.run()
else:
    stg.output(FUNI.clip_cut)
    stg.output(BILI.clip_cut)
    stg.output(filtered)
