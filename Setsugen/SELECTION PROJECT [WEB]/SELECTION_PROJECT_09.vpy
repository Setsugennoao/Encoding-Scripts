import numpy as np
import stgfunc as stg
import lvsfunc as lvf
import EoEfunc as eoe
import havsfunc as haf
from vsdpir import DPIR
from vsutil import get_y
import vapoursynth as vs
from vardefunc.misc import merge_chroma
from stgfunc.utils import replace_squaremask
from vardautomation import FileInfo, PresetAAC, PresetWEB
from selection_project_commons import SeleProFiltering, Encoding

core = vs.core

FUNI = FileInfo(
    r".\Source\[SubsPlease] Selection Project - 09 (1080p) [7EC82FDF].mkv", (240, 0), preset=(PresetWEB, PresetAAC)
)
BILI = FileInfo(
    r".\Source\[NC-Raws] SELECTION PROJECT - 09 [B-Global][WEB-DL][1080p][AVC AAC][Multiple Subtitle][MKV].mkv",
    preset=(PresetWEB)
)

OPSTART, OPENDIN = 1176, 3332


class Filtering(SeleProFiltering):
    def mix_sources(self):
        # MUSE Asia signs
        self.BILI.clip_cut = replace_squaremask(
            self.BILI.clip_cut, self.FUNI.clip_cut, (360, 150, 0, 930), [(3222, 3332), (32217, 32344)])

        # Bruuhhh
        self.BILI.clip_cut = lvf.rfs(self.BILI.clip_cut, self.FUNI.clip_cut, [
            (OPSTART + 378, OPSTART + 494), (OPSTART + 1197, OPSTART + 1208)
        ])

        # Funi shitting its pants as always
        self.FUNI.clip_cut = lvf.rfs(self.FUNI.clip_cut, self.BILI.clip_cut, [
            OPSTART + 344, (OPSTART + 346, OPSTART + 364),
            (OPSTART + 995, OPSTART + 1038), (OPSTART + 1183, OPSTART + 1186),
            OPSTART + 1196, OPSTART + 1159, OPSTART + 1303,
            # HOLY SHIT HOW IS THIS EVEN POSSIBLE, FUNIMATION
            # https://slow.pics/c/nCMAMQ9y
            (8177, 8178)
        ])

        # stg.output(lvf.diff(self.FUNI.clip_cut, self.BILI.clip_cut))

    def contras(self, clip: vs.VideoNode, ref: vs.VideoNode) -> vs.VideoNode:
        deb_y = get_y(clip)

        all_matrices = [[x, x / 2, x / 4] for x in np.arange(1, 2 ** 10)]
        error = [sum([abs(x - round(x)) for x in matrix[1:]]) for matrix in all_matrices]
        matrix = [round(all_matrices[error.index(min(error))][x]) for x in (2, 1, 2, 1, 0, 1, 2, 1, 2)]

        contra_y = core.std.Expr([
            eoe.misc.ContraSharpening(deb_y, get_y(ref)),
            deb_y.std.MergeDiff(deb_y.std.MakeDiff(deb_y.std.Convolution(matrix))),
        ], 'x dup + x y min max x y max min')

        contra_y = haf.FastLineDarkenMOD(contra_y, 22, 4, 185, 5, 1)

        return merge_chroma(contra_y, clip)

    def custom_scenefiltering(self, denoise, merge):
        deblock = DPIR(
            core.resize.Bicubic(self.BILI.clip_cut.grain.Add(0.42, 0.05), format=vs.RGBS, matrix_in=1),
            22, "deblock"
        ).resize.Bicubic(format=denoise.format.id, matrix=1)

        cs_mask = core.std.Expr([deblock, self.FUNI.clip_cut], 'x y - abs')

        merge_dcs = core.std.MaskedMerge(merge, deblock, cs_mask)

        cs = self.contras(deblock, merge_dcs)

        return lvf.rfs(denoise, cs, [(19152, 19306)])


chain = Filtering(FUNI, BILI, ((OPSTART, OPENDIN), (30426, 32344)), True)
filtered = chain.filterchain()

if __name__ == '__main__':
    brrrr = Encoding(FUNI, filtered[16315:], 1)
    brrrr.run()
else:
    stg.output(FUNI.clip_cut)
    stg.output(BILI.clip_cut)
    stg.output(filtered)
