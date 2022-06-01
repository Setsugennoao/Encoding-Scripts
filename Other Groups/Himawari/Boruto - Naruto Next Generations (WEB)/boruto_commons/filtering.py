import stgfunc as stg
import lvsfunc as lvf
import EoEfunc as eoe
import vapoursynth as vs
from vsutil import get_y, depth
from typing import Literal, Tuple
from lvsfunc.kernels import Robidoux
from stgfunc.tweaking import bbmod_fast
from fine_dehalo import contrasharpening
from vardefunc.scale import fsrcnnx_upscale
from .constants import desc_w, desc_h, kernel
from vsdenoise import knl_means_cl, ChannelMode
from debandshit import placebo_deband, dumb3kdb

core = vs.core


def bil_downscale(clip: vs.VideoNode) -> vs.VideoNode:
    clip = bbmod_fast(clip, 3, 3)
    return clip.resize.Bilinear(1280, 720, format=vs.YUV444P16)


def descale_denoiseY_filter(clip: vs.VideoNode) -> Tuple[vs.VideoNode, vs.VideoNode]:
    clip = bbmod_fast(clip, 3, 3)
    src_y = get_y(clip)

    clip_y = eoe.denoise.BM3D(src_y, 3.55, 1, "high")
    clip_y = contrasharpening(clip_y, src_y)
    clip_y = depth(clip_y, 32)

    descaled = kernel.descale(clip_y, desc_w, desc_h)
    krescaled = kernel.scale(descaled, 1920, 1080)

    descale_mask = lvf.scale.descale_detail_mask(clip_y, krescaled)

    upscaled = fsrcnnx_upscale(descaled, desc_w * 2, desc_h * 2, stg.misc.x56_SHADERS)

    rescaled = Robidoux().scale(upscaled, 1920, 1080)

    rescaled = core.std.MaskedMerge(rescaled, clip_y, descale_mask)

    return rescaled, descale_mask


def knl_filter(clip: vs.VideoNode, planes: Literal['Y', 'UV']) -> vs.VideoNode:
    sigma = 1.15 if planes == 'Y' else [None, 1.215]

    return contrasharpening(
        knl_means_cl(
            clip, sigma, 2, 2, 4, channels=ChannelMode.CHROMA
            if planes == 'UV' else ChannelMode.LUMA),
        clip)


def debanding_filter(clip: vs.VideoNode) -> vs.VideoNode:
    return core.f3kdb.Deband(
        core.average.Mean([
            dumb3kdb(clip, 8, 25),
            placebo_deband(clip, 16, 3, 1, 0),
            placebo_deband(clip, 24, 4, 1, 0),
        ]), 16, 0, 0, 0, 16, 16
    )


def limit_yuv_filter(clip: vs.VideoNode) -> vs.VideoNode:
    return clip.std.Limiter(16 << 2, [235 << 2, 240 << 2], [0, 1, 2])


def adaptive_grain(clip: vs.VideoNode, strength=0.25, luma_scaling=12) -> vs.VideoNode:
    mask = core.adg.Mask(clip.std.PlaneStats(), luma_scaling)
    grained = core.grain.Add(clip, var=strength, constant=True)

    return core.std.MaskedMerge(clip, grained, mask)
