import numpy as np
import lvsfunc as lvf
import EoEfunc as eoe
import vapoursynth as vs
from typing import Tuple
from vardefunc import merge_chroma
from lvsfunc.kernels import Spline64
from vsutil import plane, insert_clip
from stgfunc.tweaking import bbmod_fast
from fine_dehalo import contrasharpening
from lvsfunc.scale import ssim_downsample
from vsdenoise import knl_means_cl, ChannelMode

from .classes import Crop, SimpleCrop
from .masking import get_downdescale_mask
from .cropping import crop_resize_to_720p


core = vs.core

ENDING_15 = [
    SimpleCrop((0, 756)),
    SimpleCrop((767, 938)),
    SimpleCrop((948, 1137)),
    SimpleCrop((1278, 1662)),
    SimpleCrop((1850, 2144)),
    Crop((1663, 1777), 314, 315, 0, 0, False),
    Crop((1778, 1849), 257, 255),
    Crop((1138, 1184), 226, 244, 141, 638),
    Crop((1185, 1229), 211, 466, 135, 136),
    Crop((1230, 1277), 170, 146, 816, 87),
]


def filterEnding15(clip: vs.VideoNode, ED_FRAMES: Tuple[int, int]) -> vs.VideoNode:

    def _filterEnding(n: int, f: vs.VideoFrame) -> vs.VideoNode:
        ed_n = n - ED_FRAMES[0]
        if n not in np.arange(ED_FRAMES[0], ED_FRAMES[1] + 1):
            return clip[n]

        for i in np.arange(len(ENDING_15)):
            crop = ENDING_15[i]
            if ed_n in np.arange(crop.frames[0], crop.frames[1] + 1):
                return crop_resize_to_720p(clip, crop)[n]

        return clip[n]

    return core.std.FrameEval(clip, _filterEnding, clip)


def filterEnding16(src: vs.VideoNode, clip: vs.VideoNode, ED_FRAMES: Tuple[int, int]) -> vs.VideoNode:
    ending = src[ED_FRAMES[0]:ED_FRAMES[1] + 1]

    clip_y = plane(ending, 0)
    denoise_y = eoe.denoise.BM3D(clip_y, 1.45, 1, 'high')

    denoised = merge_chroma(
        contrasharpening(knl_means_cl(denoise_y, 1.54, 1, 2, 4, channels=ChannelMode.LUMA), clip_y),
        contrasharpening(knl_means_cl(ending, [1.14, 1.32], 1, 2, 8, channels=ChannelMode.CHROMA), ending)
    )

    edfx = denoised.edgefixer.ContinuityFixer(*(4,) * 4)
    edfx = denoised.edgefixer.ContinuityFixer(*(3,) * 4)

    ed_final = edfx.resize.Spline64(1280, 720, format=vs.YUV444P16)

    return insert_clip(clip, ed_final, ED_FRAMES[0])


def filterEnding17(src: vs.VideoNode, clip: vs.VideoNode, ED_FRAMES: Tuple[int, int]) -> vs.VideoNode:
    ending = src[ED_FRAMES[0]:ED_FRAMES[1] + 1]
    yuv_444 = ending.resize.Spline64(1280, 720, format=vs.YUV444P16)

    clip_y = plane(ending, 0)
    denoise_y = eoe.denoise.BM3D(clip_y, 1.35, 1, 'high')

    denoise_y = ssim_downsample(denoise_y, 1280, 720, kernel=Spline64(format=yuv_444.format.id))

    denoise_uv = merge_chroma(
        denoise_y, contrasharpening(knl_means_cl(
            denoise_y, [1.14, 1.32], 1, 2, 8, channels=ChannelMode.CHROMA
        ), yuv_444)
    )

    denoised = core.std.ShufflePlanes([denoise_y, denoise_uv], [0, 1, 2], vs.YUV)

    edgefixed = bbmod_fast(denoised, 1, 1)

    return insert_clip(clip, edgefixed, ED_FRAMES[0])


def filterEnding18(src: vs.VideoNode, clip: vs.VideoNode, ED_FRAMES: Tuple[int, int]) -> vs.VideoNode:
    ending = src[ED_FRAMES[0]:ED_FRAMES[1] + 1]
    yuv_444 = ssim_downsample(ending, 1280, 720, kernel=Spline64(format=vs.YUV444P16))

    denoise_y = eoe.denoise.BM3D(yuv_444, 0.45, 1, 'high', chroma=False)

    denoised = merge_chroma(denoise_y, contrasharpening(knl_means_cl(
        denoise_y, 0.84, 1, 2, 8, channels=ChannelMode.CHROMA), yuv_444))

    edgefixed = bbmod_fast(denoised, 1, 1, 1, 1)
    edgefixed = bbmod_fast(edgefixed, 2, 2, 2, 2)

    return insert_clip(clip, edgefixed, ED_FRAMES[0])


def filterTVTokyo(clip: vs.VideoNode, bil_downscale: vs.VideoNode, TV_TOKYO_FRAMES: Tuple[int, int]) -> vs.VideoNode:
    s = TV_TOKYO_FRAMES[0]

    tvtokyo = bil_downscale[s + 17] + bil_downscale[s + 17: s + 37]

    tvtokyo_filt = lvf.deblock.vsdpir(tvtokyo, 5, 'denoise', 1, i444=True)

    tvtokyo_filt = (tvtokyo_filt[0] * 17) + tvtokyo_filt[1:-1] + (tvtokyo_filt[-1] * (TV_TOKYO_FRAMES[1] - 37))

    return insert_clip(clip, tvtokyo_filt, TV_TOKYO_FRAMES[0])


def filterPreview(
    clip: vs.VideoNode, replace: vs.VideoNode, descale_mask: vs.VideoNode, ED_FRAMES: Tuple[int, int]
) -> vs.VideoNode:
    downscaled_descale_mask = get_downdescale_mask(descale_mask)

    bil_masked = core.std.MaskedMerge(clip, replace, downscaled_descale_mask)

    return lvf.rfs(clip, bil_masked, (ED_FRAMES[1] + 1, clip.num_frames))
