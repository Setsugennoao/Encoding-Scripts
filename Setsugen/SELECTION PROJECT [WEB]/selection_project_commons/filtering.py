import stgfunc as stg
import lvsfunc as lvf
import EoEfunc as eoe
from vsdpir import DPIR
import vapoursynth as vs
from stgfunc import depth
from vsutil import insert_clip
from debandshit import dumb3kdb
from lvsfunc.util import get_prop
from vardautomation import FileInfo
from vardautomation import PresetWEB
from vardefunc.noise import Graigasm
from .constants import graigasm_args
from vardefunc.misc import merge_chroma
from vardefunc.util import finalise_output
from stgfunc.utils import replace_squaremask
from typing import Tuple, List, Dict, Optional
from vsutil import get_y, join, plane, iterate
from vsdenoise.knlm import knl_means_cl, ChannelMode

core = vs.core

core.max_cache_size = 16 * 2 ** 10


class SeleProFiltering:
    FUNI: FileInfo
    BILI: FileInfo
    lowpass: List[int]
    Oycore: stg.oyster.Core
    dfttest_args: Dict[str, int]

    def __init__(
        self, FUNI: FileInfo, BILI: FileInfo,
        OP_ED: Optional[Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]] = None,
        OP_START_REPLACE: bool = False
    ):
        self.FUNI = FUNI
        self.BILI = BILI
        self.OP_ED = OP_ED
        self.OP_START_REPLACE = OP_START_REPLACE

    @finalise_output()
    def workraw_filterchain(self):
        src = depth(self.FUNI.clip_cut, 16)

        ref = src.rgvs.RemoveGrain(16)

        denoise = eoe.dn.BM3D(src, 1.95, 1, 'fast', fast=True, ref=ref, skip_basic=True, chroma=False)

        deband = dumb3kdb(denoise, 8, 24)

        return deband.add.Grain(0.25)

    @finalise_output()
    def filterchain(self):
        self.mix_sources()

        if self.OP_ED:
            self.mix_OP_ED(self.FUNI)
            self.mix_OP_ED(self.BILI)

        y = depth(get_y(self.FUNI.clip_cut), 16)

        self.__setup_oyster(y, plane(self.FUNI.clip_cut, 1))

        y1, y2 = self.__freqmerging_luma(y)
        u, v = self.__freqmerging_chroma()

        diff_mask = self.__masks_diff(y1, y2)

        detail_mask = self.__masks_detail(y)

        linemask = self.__masks_linemask(y2)

        exp_linemask = self.__masks_exp_linemask(linemask)

        detail_mask2 = self.__masks_detail2(detail_mask, exp_linemask, diff_mask)

        details = y1.std.MaskedMerge(y, detail_mask2)

        y_masked = details.std.MaskedMerge(y2, linemask)

        merge = join([y_masked, u, v])

        denoise = self.__denoise(merge)

        if self.OP_ED:
            denoise = self.__scenefilter_OP_ED(denoise)

        custom = self.custom_scenefiltering(denoise, merge)

        grain = self.__graining(custom, detail_mask2)

        return grain.std.MaskedMerge(custom, linemask)

    def mix_sources(self) -> None:
        pass

    def mix_OP_ED(self, file: FileInfo) -> None:
        if self.OP_ED[0]:
            OP_AV1 = stg.src(r".\Extra\NCs\Source\SELECTION PROJECT OPテーマ 「Glorious Days」_AV1.mp4",
                             ref=file.clip_cut)[:2158]
            OP_START, OP_ENDIN = self.OP_ED[0]

            texture = OP_AV1.grain.Add(20, 3, 0.07, 0.12, 69420, True).bilateral.Gaussian(1)

            merge = file.clip_cut[OP_START:OP_ENDIN + 1]

            if self.OP_START_REPLACE:
                FUNI03 = FileInfo(
                    r".\Source\[SubsPlease] Selection Project - 03 (1080p) [4C3303CD].mkv", (240, 0),
                    preset=PresetWEB)

                OP_EP03 = FUNI03.clip_cut[480:2637 + 1][0:195 + 1]

                merge = insert_clip(merge, OP_EP03, 0)

            deband = OP_AV1.bilateral.Gaussian(1.5)
            merge = replace_squaremask(merge, deband, (727, 50, 599, 516), (None, 195))

            merge = replace_squaremask(merge, texture, (993, 50, 464, 516), (2042, None))
            merge = replace_squaremask(merge, OP_AV1, (624, 50, 833, 516), (2042, None))

            black = merge.std.BlankClip(length=1)
            white = black.std.Invert()

            merge = black + merge[1:195] + white + merge[196:2041] + white + merge[2042:]

            file.clip_cut = insert_clip(file.clip_cut, merge, OP_START)
            file.clip_cut = lvf.rfs(file.clip_cut, self.FUNI.clip_cut, (OP_START, OP_START + 56))

        if self.OP_ED[1]:
            ED_START, ED_ENDIN = self.OP_ED[1]
            ED_VP9 = stg.src(r".\Extra\NCs\Source\SELECTION PROJECT EDテーマ 「Only One Yell」_VP9.webm", ref=file.clip_cut)

            ED_CUT = file.clip_cut[ED_START:ED_ENDIN + 1]

            while ED_VP9.num_frames == ED_CUT.num_frames:
                ED_VP9 += ED_VP9[-1]

            merge = replace_squaremask(ED_CUT, ED_VP9, (993, 50, 464, 516), (1791, None))
            file.clip_cut = insert_clip(file.clip_cut, merge, ED_START)

    def custom_scenefiltering(self, denoise: vs.VideoNode, merge: vs.VideoNode):
        return denoise

    def __setup_oyster(self, y: vs.VideoNode, chroma: vs.VideoNode) -> None:
        self.Oycore = stg.oyster.Core()
        self.lowpass = [0.0, 0.0, 0.12, 1024.0, 1.0, 1024.0]
        self.dfttest_args = dict(smode=0, sosize=0, tbsize=1, tosize=0, tmode=0)

        self.block_mask = depth(self.Oycore.GenBlockMask(y), 32)
        self.block_mask_uv = depth(self.Oycore.GenBlockMask(chroma), 32)

    def __freqmerging_luma(
        self, funi: vs.VideoNode = None, bili: vs.VideoNode = None
    ) -> Tuple[vs.VideoNode, vs.VideoNode]:
        funi_y, bili_y = depth(get_y(funi or self.FUNI.clip_cut), get_y(bili or self.BILI.clip_cut), 32)

        funi_y_ref = stg.oyster.Basic(funi_y, None, 6, 1, 2400, True)
        bili_y_ref = stg.oyster.Basic(bili_y, None, 6, 1, 2800, True)

        args = dict(sbsize=9, slocation=self.lowpass, **self.dfttest_args)

        funif = core.dfttest.DFTTest(funi_y, **args)
        bilif = core.dfttest.DFTTest(bili_y, **args)

        funi_y_reff = core.dfttest.DFTTest(funi_y_ref, **args)
        bili_y_reff = core.dfttest.DFTTest(bili_y_ref, **args)

        funi_mer_y = core.std.MergeDiff(bilif, core.std.MakeDiff(funi_y, funif))
        bili_mer_y = core.std.MergeDiff(funif, core.std.MakeDiff(bili_y, bilif))

        funi_y_ref_mer = core.std.MergeDiff(bili_y_reff, core.std.MakeDiff(funif, funi_y_reff))
        bili_y_ref_mer = core.std.MergeDiff(funi_y_reff, core.std.MakeDiff(bilif, bili_y_reff))

        freqmerged_funi = core.std.MaskedMerge(
            funi_mer_y, funi_y_ref_mer, self.block_mask
        ).std.Merge(funi_mer_y, 4 / 7)

        freqmerged_bili = core.std.MaskedMerge(
            bili_mer_y, bili_y_ref_mer, self.block_mask
        ).std.Merge(funi_y_ref_mer, 2 / 5)  # Intentional Funi here

        return depth(freqmerged_funi, freqmerged_bili, 16)

    def ___freqmerge_chroma(self, clip_y: vs.VideoNode, filt_y: vs.VideoNode) -> vs.VideoNode:
        ref = stg.oyster.Basic(clip_y, None, 6, 1, True)
        ref = self.Oycore.FreqMerge(filt_y, ref, 9, self.lowpass)

        mer_y = self.Oycore.FreqMerge(filt_y, clip_y, 9, self.lowpass)

        freqmerged = core.std.MaskedMerge(
            mer_y, ref, self.block_mask_uv
        ).std.Merge(ref, 3 / 5)

        return depth(freqmerged, 16)

    def __freqmerging_chroma(
        self, funi: vs.VideoNode = None, bili: vs.VideoNode = None
    ) -> Tuple[vs.VideoNode, vs.VideoNode]:
        funi, bili = funi or self.FUNI.clip_cut, bili or self.BILI.clip_cut

        ub, uf = depth(plane(bili, 1), plane(funi, 1), 32)
        vb, vf = depth(plane(bili, 2), plane(funi, 2), 32)

        return self.___freqmerge_chroma(ub, uf), self.___freqmerge_chroma(vb, vf)

    def __masks_diff(self, y1: vs.VideoNode, y2: vs.VideoNode) -> vs.VideoNode:
        _diff_mask = core.std.MakeDiff(y1, y2).std.PlaneStats()

        black = _diff_mask.std.BlankClip()[0].get_frame(0)

        thr = 100 << 8

        def ___diff_mask_process(f, n) -> vs.VideoNode:
            low_range = get_prop(f, 'PlaneStatsMin', int) <= thr
            hig_range = get_prop(f, 'PlaneStatsMax', int) >= (2 << 16) - 1 - thr > thr

            return f if low_range or hig_range else black

        diff_mask = _diff_mask.std.ModifyFrame(
            _diff_mask, ___diff_mask_process
        )

        return diff_mask.std.BinarizeMask(135 << 8).std.Inflate()

    def __masks_detail(self, y: vs.VideoNode) -> vs.VideoNode:
        return stg.mask.linemask(y)

    def __masks_linemask(self, y: vs.VideoNode) -> vs.VideoNode:
        return stg.mask.tcanny(y, 0.0275, True).std.Inflate()

    def __masks_exp_linemask(self, linemask: vs.VideoNode) -> vs.VideoNode:
        exp_linemask = linemask.std.BinarizeMask(24 << 8)

        exp_linemask = iterate(
            iterate(
                exp_linemask,
                lambda x: x.std.Maximum().std.Minimum(), 10
            ), lambda x: x.std.Inflate().std.Deflate(), 5
        )

        exp_linemask = exp_linemask.bilateral.Gaussian(1.5, 1)

        return iterate(exp_linemask, core.std.Deflate, 15)

    def __masks_linemask3(
            self, linemask: vs.VideoNode, exp_linemask: vs.VideoNode, diff_mask: vs.VideoNode) -> vs.VideoNode:
        return core.std.Expr([linemask, exp_linemask, diff_mask], 'y x 2 * - 4 * z +').std.Limiter()

    def __masks_detail2(
            self, details_mask: vs.VideoNode, exp_linemask: vs.VideoNode, diff_mask: vs.VideoNode) -> vs.VideoNode:
        return core.std.Expr([details_mask, exp_linemask, diff_mask], 'x y - z +').std.Limiter()

    def __masks_deband(
        self, linemask: vs.VideoNode, exp_linemask: vs.VideoNode, linemask3: vs.VideoNode, diff_mask: vs.VideoNode
    ) -> vs.VideoNode:
        return core.std.Expr([
            core.std.Expr([
                linemask, exp_linemask, linemask3, diff_mask
            ], 'y x 2 * - 4 * z log - a +'), linemask3
        ], 'x y -').std.Limiter()

    def __denoise(self, clip: vs.VideoNode) -> vs.VideoNode:
        denoise_bm3d = eoe.dn.BM3D(clip, 1.75, 1, ['np', 'high'], chroma=False)

        return knl_means_cl(denoise_bm3d, 0.37, channels=ChannelMode.CHROMA)

    def __scenefilter_OP_ED(self, clip: vs.VideoNode) -> vs.VideoNode:
        if self.OP_ED[0]:
            OP_START, OP_ENDIN = self.OP_ED[0]

            deband = dumb3kdb(
                DPIR(
                    depth(clip.resize.Spline64(format=vs.RGB24, matrix_in=1), 32), 4.35
                ).resize.Spline64(format=clip.format.id, matrix=1), 16, 12
            )

            return lvf.rfs(clip, deband, (OP_START + 1, OP_START + 56))
        return clip

    def __graining(self, clip: vs.VideoNode, detail_mask: vs.VideoNode) -> vs.VideoNode:
        grain_mask = core.adg.Mask(clip.std.PlaneStats(), 8)

        grain_mask = core.std.Expr([grain_mask, detail_mask], 'x y -')

        y = get_y(clip)

        pref = iterate(y, core.std.Maximum, 2).std.Convolution([1] * 9)

        grainY = Graigasm(**graigasm_args).graining(y, prefilter=pref)

        return merge_chroma(grainY, clip)
