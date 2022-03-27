import stgfunc as stg
import lvsfunc as lvf
import vapoursynth as vs
from stgfunc.utils import replace_squaremask
from selection_project_commons import SeleProFiltering, Encoding
from vardautomation import FileInfo, PresetAAC, PresetWEB, get_vs_core

core = get_vs_core(range(0, vs.core.num_threads, 2))

FUNI = FileInfo(
    r".\Source\[SubsPlease] Selection Project - 11 (1080p) [A50CC4E9].mkv", (240, 0), preset=(PresetWEB, PresetAAC)
)
BILI = FileInfo(
    r".\Source\[NC-Raws] SELECTION PROJECT - 11 [B-Global][WEB-DL][1080p][AVC AAC][Multiple Subtitle][MKV].mkv",
    preset=(PresetWEB)
)

OPSTART, OPENDIN = 624, 2781


class Filtering(SeleProFiltering):
    def mix_sources(self):
        # MUSE Asia signs
        self.BILI.clip_cut = replace_squaremask(
            self.BILI.clip_cut, self.FUNI.clip_cut, (360, 150, 0, 930), [(2671, 2781), (33513, 33639)]
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


chain = Filtering(FUNI, BILI, ((OPSTART, OPENDIN), (31722, 33639)), True)
filtered = chain.filterchain()

if __name__ == '__main__':
    brrrr = Encoding(FUNI, filtered)
    brrrr.run()
else:
    stg.output(FUNI.clip_cut)
    stg.output(BILI.clip_cut)
    stg.output(filtered)
