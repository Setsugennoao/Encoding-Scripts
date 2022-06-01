import stgfunc as stg
import lvsfunc as lvf
from vsdpir import DPIR
import vapoursynth as vs
from stgfunc import depth
from stgfunc.utils import replace_squaremask
from vardautomation import FileInfo, PresetAAC, PresetWEB
from selection_project_commons import SeleProFiltering, Encoding

core = vs.core

FUNI = FileInfo(
    r".\Source\[SubsPlease] Selection Project - 06 (1080p) [7943CAF4].mkv", (240, 0), preset=(PresetWEB, PresetAAC)
)
BILI = FileInfo(
    r".\Source\[NC-Raws] SELECTION PROJECT - 06 [B-Global][WEB-DL][1080p][AVC AAC][Multiple Subtitle][MKV].mkv",
    preset=(PresetWEB)
)

OPSTART, OPENDIN = 2710, 4866


class Filtering(SeleProFiltering):
    def mix_sources(self):
        # MUSE Asia signs
        self.BILI.clip_cut = replace_squaremask(
            self.BILI.clip_cut, self.FUNI.clip_cut, (360, 150, 0, 930), [(4756, 4866), (33343, 33469)]
        )

        # Bruuhhh
        self.BILI.clip_cut = lvf.rfs(self.BILI.clip_cut, self.FUNI.clip_cut, [
            (OPSTART + 378, OPSTART + 494), (OPSTART + 1197, OPSTART + 1208)
        ])

        # Funi shitting its pants as always
        self.FUNI.clip_cut = lvf.rfs(self.FUNI.clip_cut, self.BILI.clip_cut, [
            OPSTART + 344, (OPSTART + 346, OPSTART + 364),
            (OPSTART + 995, OPSTART + 1038), (OPSTART + 1183, OPSTART + 1186),
            OPSTART + 1196, OPSTART + 1159, OPSTART + 1303, (2527, 2528),
            (0, 431)
        ])

        # stg.output(lvf.diff(self.FUNI.clip_cut, self.BILI.clip_cut))

    def custom_scenefiltering(self, denoise, merge):
        rgb_32 = depth(merge.resize.Spline64(format=vs.RGB24, matrix_in=1), 32)
        rgb_deb_dpir = DPIR(DPIR(rgb_32, 1.18), 10, 'deblock')
        deb_dpir = rgb_deb_dpir.resize.Spline64(format=denoise.format.id, matrix=1)

        return lvf.rfs(denoise, deb_dpir, (0, 431))


chain = Filtering(FUNI, BILI, ((OPSTART, OPENDIN), (31552, 33469)), True)
filtered = chain.filterchain()

if __name__ == '__main__':
    brrrr = Encoding(FUNI, filtered)
    brrrr.run()
else:
    stg.output(FUNI.clip_cut)
    stg.output(BILI.clip_cut)
    stg.output(filtered)
