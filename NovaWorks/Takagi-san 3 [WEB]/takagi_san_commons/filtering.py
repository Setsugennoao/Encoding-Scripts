from ccd import ccd
import stgfunc as stg
import lvsfunc as lvf
from typing import List
import vapoursynth as vs
from stgfunc import depth
from debandshit import dumb3kdb
from vsmask.edge import PrewittStd
from fine_dehalo import fine_dehalo
from vardefunc.misc import merge_chroma
from vsutil import get_y, get_w, iterate
from lvsfunc.scale import ssim_downsample
from vardefunc.util import finalise_output
from lvsfunc.kernels import Catrom, Robidoux
from vardefunc.noise import Graigasm, AddGrain
from vardefunc.aa import Eedi3SR, upscaled_sraa
from vardefunc.scale import to_444, fsrcnnx_upscale

from .masking import linemask, detail_mask


core = vs.core
catrom = Catrom()


class TakagiSanSanFiltering:
    muse_asia_crap_ranges = []
    no_rescale_ranges = []

    def __init__(self):
        pass

    @finalise_output
    def filtering(
        self, sources: vs.VideoNode, weights: List[float] = [0.55, 0.63, 0.40, 0.72]
    ) -> vs.VideoNode:
        amazon_cbr, amazon_vbr, bili, netflix = [to_444(clip, None, None, True) for clip in sources]

        if len(self.muse_asia_crap_ranges):
            bili = lvf.rfs(bili, core.average.Mean([amazon_cbr, amazon_vbr, netflix]), self.muse_asia_crap_ranges)

        average = core.average.Mean([amazon_cbr, amazon_vbr, bili])

        netflix = core.average.Mean([
            netflix.deblock.Deblock(quant) for quant in range(16, 32, 4)
        ] + [netflix.rgsf.Repair(average, 4)])

        median = core.median.Median([amazon_cbr, amazon_vbr, bili, netflix, average])

        lineart = linemask(bili)

        lineart_mask = PrewittStd().edgemask(lineart, 0.5)

        ref_clips = [
            clip.std.MaskedMerge(bili, lineart_mask)
            for clip in [amazon_cbr, amazon_vbr, netflix, median]
        ] + [average, bili]

        weights += [0.80, 0.70]

        weighted_average = stg.utils.weighted_merge(*zip(ref_clips, weights))

        rescaled = self.rescale(weighted_average)

        chromafix = rescaled.warp.AWarpSharp2(thresh=92, blur=3, type=1, depth=4, planes=[1, 2])
        chromafix = ccd(chromafix, 5)
        chromafix = Catrom().resample(chromafix, vs.YUV420P16)

        dehalo = fine_dehalo(chromafix, None, 2.1, 2.1, darkstr=0, highsens=85)

        deband = self.debanding(dehalo)

        return self.graining(deband)

    def rescale(self, clip: vs.VideoNode) -> vs.VideoNode:
        y = get_y(clip)

        descaled = lvf.scale.descale(y, None, None, range(764, 769), Robidoux())
        descaled = catrom.scale(descaled, get_w(769), 769)
        descaled = depth(descaled, 16)

        upscale = fsrcnnx_upscale(descaled, None, 6969, stg.misc.x56_SHADERS, None, strength=70)
        upscale_no_descale = fsrcnnx_upscale(
            y, upscale.width, upscale.height, stg.misc.x56_SHADERS, ssim_downsample, strength=55
        )
        upscale = lvf.rfs(upscale, upscale_no_descale, self.no_rescale_ranges)

        eedaa = Eedi3SR(eedi3cl=True, alpha=0.2, beta=0.5, gamma=500, nrad=3, mdis=15).do_aa()(upscale)

        up_sraa = upscaled_sraa(descaled, 3, eedaa.width, eedaa.height)
        up_sraa = lvf.rfs(up_sraa, upscale_no_descale, self.no_rescale_ranges)

        aaa = lvf.aa.clamp_aa(upscale, eedaa, up_sraa, 0.85)

        down = catrom.scale(aaa, clip.width, clip.height)

        yuv444 = catrom.resample(down, vs.YUV444P16)

        based = lvf.aa.based_aa(yuv444, stg.misc.x56_SHADERS)

        rescaled = catrom.resample(based, vs.GRAY16)
        return merge_chroma(rescaled, clip)

    def debanding(self, clip: vs.VideoNode) -> vs.VideoNode:
        deband_mask = detail_mask(clip, brz=(1200, 3750))
        deband_mask2 = core.std.Expr([
            core.adg.Mask(get_y(clip).std.PlaneStats(), 26), deband_mask
        ], 'x y -')

        deband1 = core.std.MaskedMerge(dumb3kdb(clip, 8, 24, [12, 4]), clip, deband_mask)
        return core.std.MaskedMerge(deband1, dumb3kdb(clip, 16, 30, [6, 0]), deband_mask2)

    def graining(self, clip: vs.VideoNode) -> vs.VideoNode:
        pref = iterate(get_y(clip), core.std.Maximum, 2).std.Convolution([1] * 9)

        return Graigasm(
            thrs=[x << 8 for x in (58, 80, 128)],
            strengths=[(0.64, 0.15), (0.35, 0.1), (0.14, 0.06)],
            sizes=(1.2, 1.16, 1.1),
            sharps=(55, 45, 40),
            grainers=[
                AddGrain(seed=80085, constant=False),
                AddGrain(seed=69420, constant=True),
                AddGrain(seed=69420, constant=True)
            ]
        ).graining(clip, prefilter=pref)
