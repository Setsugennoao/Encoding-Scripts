import adjust
import stgfunc as stg
import lvsfunc as lvf
import EoEfunc as eoe
import kagefunc as kgf
from vsdpir import DPIR
import vapoursynth as vs
from stgfunc import depth
from debandshit import dumb3kdb
from vsutil import get_y, insert_clip
from stgfunc.utils import replace_squaremask
from vardautomation import FileInfo, PresetAAC, PresetWEB
from selection_project_commons import SeleProFiltering, Encoding

core = vs.core

FUNI = FileInfo(
    r".\Source\[SubsPlease] Selection Project - 03 (1080p) [4C3303CD].mkv", (240, 0), preset=(PresetWEB, PresetAAC)
)
BILI = FileInfo(
    r".\Source\[NC-Raws] SELECTION PROJECT - 03 [B-Global][WEB-DL][1080p][AVC AAC][Multiple Subtitle][MKV].mkv",
    preset=(PresetWEB)
)

OPSTART, OPENDIN = 480, 2637


class Filtering(SeleProFiltering):
    def mix_sources(self):
        # MUSE Asia signs
        self.BILI.clip_cut = replace_squaremask(
            self.BILI.clip_cut, self.FUNI.clip_cut, (360, 150, 0, 930), [(2526, 2637), (32720, 32847)]
        )

        # Bruuhhh
        self.BILI.clip_cut = lvf.rfs(self.BILI.clip_cut, self.FUNI.clip_cut, [
            (OPSTART + 378, OPSTART + 494), (OPSTART + 1197, OPSTART + 1208), (28159, 28162), (28563, 28579)
        ])

        # Funi shitting its pants as always
        self.FUNI.clip_cut = lvf.rfs(self.FUNI.clip_cut, self.BILI.clip_cut, [
            OPSTART + 344, (OPSTART + 346, OPSTART + 364),
            (OPSTART + 995, OPSTART + 1038), (OPSTART + 1183, OPSTART + 1186),
            OPSTART + 1196, (28163, 28179), (28232, 28296), (28548, 28562),
            (28763, 28791), (29376, 29394), (29502, 29533), (29705, 30017)
        ])

        bili_16 = depth(self.BILI.clip_cut, 16)

        bili_denoise = eoe.dn.CMDegrain(bili_16, 2, 750, contrasharp=True)

        bili_deband = depth(dumb3kdb(bili_denoise, 8, 18), 8)

        self.BILI.clip_cut = lvf.rfs(self.BILI.clip_cut, self.FUNI.clip_cut, (20269, 20621))
        self.BILI.clip_cut = lvf.rfs(self.BILI.clip_cut, bili_deband, (20622, 21687))

        # stg.output(lvf.diff(self.FUNI.clip_cut, self.BILI.clip_cut))

    def custom_scenefiltering(self, denoise, merge):
        denoise_cut = denoise[9310:9345]

        dehardsub = denoise_cut[0] * 35
        y = get_y(denoise_cut)

        mask = core.std.BlankClip(y).sub.Subtitle(
            r"{\an7\pos(0,0)\p1}m 248 362 l 1472 363 1461 441 1462 601 1451 622 1453 644 1432 647 1419 655 1419 669 1401 682 1404 718 249 724"
        ).std.Binarize(24).std.InvertMask().bilateral.Gaussian(15)

        dehardsub = dehardsub.std.MaskedMerge(denoise_cut, mask)

        mask1 = core.std.BlankClip(y).sub.Subtitle(
            r"{\an7\blur1\bord0\shad0\fscx100\fscy100\pos(0,0)\p1}m 1426 647 l 1380 648 1394 495 1431 442 1425 552"
        ).std.Binarize(24).bilateral.Gaussian(45)

        darkened = depth(dehardsub, 32).std.MaskedMerge(*depth(adjust.Tweak(dehardsub, bright=-20), mask1, 32))

        mask2 = core.std.BlankClip(y).sub.Subtitle(
            r"{\an7\blur1\bord0\shad0\fscx110\fscy100\pos(-145,0)\p1}m 1426 647 l 1391.2 648 1424.96 612.11"
        ).std.Binarize(24).bilateral.Gaussian(1.7)

        darkened = darkened.std.MaskedMerge(*depth(adjust.Tweak(dehardsub, bright=-8.5), mask2, 32))

        mask3 = core.std.BlankClip(y).sub.Subtitle(
            r"{\an7\blur1\bord0\shad0\fscx100\fscy100\pos(0,0)\p1}m 1403.43 607.3 b 1407.91 632.6 1413.84 649.41 1426.35 663.75 1377.91 666.86 1343.09 641.21 1403.04 607.25"
        ).std.Binarize(24).bilateral.Gaussian(1.7)

        rgb_32 = darkened.resize.Spline64(format=vs.RGBS, matrix_in=1)
        dpir_dark = DPIR(rgb_32, 35).resize.Spline64(format=darkened.format.id, matrix=1)
        dehardsub = depth(darkened.std.MaskedMerge(dpir_dark, depth(
            core.std.Expr([mask, mask1, mask2, mask3], 'x y + z a + +'), 32)), 16
        )

        dehardsub = lvf.rfs(
            dehardsub, dehardsub.std.MaskedMerge(
                denoise_cut, kgf.squaremask(denoise_cut, 86, 67, 1388, 683).bilateral.Gaussian(7.5)
            ), 8
        )

        denoise = insert_clip(denoise, dehardsub, 9310)

        rgb_32 = depth(merge.resize.Spline64(format=vs.RGB24, matrix_in=1), 32)
        rgb_dpir = DPIR(DPIR(rgb_32, 3.45,), 25, 'deblock')
        dpir = rgb_dpir.resize.Spline64(format=denoise.format.id, matrix=1)

        dpir = dpir.grain.Add(0.46, 0.04, 0.065, 0.065, 69420, False)

        # Preferring to keep my sanity rather than
        # preserving some details, like in 29210 pensive smh
        return lvf.rfs(denoise, dpir, [
            (26930, 27161), (27233, 27777), (28159, 28427),
            (28547, 28868), (29128, 29233), (29294, 30017)
        ])


chain = Filtering(FUNI, BILI, ((OPSTART, OPENDIN), (30929, 32847)))
filtered = chain.filterchain()

if __name__ == '__main__':
    brrrr = Encoding(FUNI, filtered)
    brrrr.run()
else:
    stg.output(FUNI.clip_cut)
    stg.output(BILI.clip_cut)
    stg.output(filtered)
