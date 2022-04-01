import math as mathematics
import lvsfunc as lvf
import vsutil
import cooldegrain
import insaneAA
import atomchtools as atf
import fvsfunc as fvf
import kagefunc as kg
import havsfunc
import mvsfunc as mvf
import GradFun3 as gfa
import insaneAA 
import cooldegrain
import finedehalo as fd
import HardAAp2 as hiaa
import descale as dsc
import psutil
import regress
import stgfunc as setsugen_no_vapoursynth_functions
import havsfunc as haf
from vardautomation import X265, FileInfo
import vardefunc as vdf
from fine_dehalo import fine_dehalo
import vapoursynth as vs
import os
from vardefunc.scale import to_444
from vardefunc.noise import Graigasm, AddGrain
from math import ceil
from lvsfunc.kernels import Catrom
from vsdenoise.bm3d import BM3DCudaRTC, Profile
from vardefunc.util import finalise_clip
from vsutil import get_w as get_width_relative_to_height_and_aspect_ratio

from fine_dehalo import contrasharpening_fine_dehalo

core = vs.core
vardefunc = vdf


fileinfo = FileInfo(r'.\Source\Johnny Test - 01 [jp dub] (Netflix 480p).mkv')

dvd = setsugen_no_vapoursynth_functions.src(fileinfo.path, 16, matrix_prop=5, force_lsmas=True)

sizes = get_width_relative_to_height_and_aspect_ratio(576, dvd.width / dvd.height), 576

dvd_444 = to_444(dvd, None, None, True, False)
dvd_444 = dvd_444[:15985] + dvd_444[29212:]

dvd_444 = Catrom().scale(dvd_444, *sizes)
dvd_444 = Catrom().resample(dvd_444, vs.YUV444P16)


def _getasset(assetname: str, length: int) -> vs.VideoNode:
    asset = setsugen_no_vapoursynth_functions.src(rf".\assets\{assetname}.png")
    resized = asset.resize.Bicubic(*sizes, vs.YUV444P16, matrix=5).std.Loop(length)
    return resized.std.AssumeFPS(dvd_444)


def _ms_to_frames(ms: int) -> int:
    return ceil(ms * (24 / 1001))


# we do a little trolling
dvddddddd = core.std.Splice([
    dvd_444[:1462],
    _getasset('2_intro_sponsor_card', _ms_to_frames(10024)),
    dvd_444[1462:9337],
    _getasset('4_eyecatch_A', _ms_to_frames(4004)),
    _getasset('5_eyecatch_B', _ms_to_frames(4004)),
    dvd_444[9337:15792],
    _getasset('8_ending', 733),
    _getasset('9_outro_sponsor_card', _ms_to_frames(10024)),
    _getasset('10_outro_card_5s', _ms_to_frames(5000)),
], False)

den = BM3DCudaRTC(dvddddddd, 8, 1).clip.std.RemoveFrameProps('_Matrix')

ref = den.bilateral.Bilateral(sigmaS=1.5, sigmaR=7/255)
bidh = den.bilateral.Bilateral(ref=ref, sigmaS=1.5 / 3, sigmaR=5/255)

dehalo_merge = core.std.Expr([dvddddddd, bidh], "x y min")
dehalo_merge = contrasharpening_fine_dehalo(dehalo_merge, dvddddddd, 1.5)
dehalo_merge = fine_dehalo(dehalo_merge, darkstr=0)

denoise = BM3DCudaRTC(dehalo_merge, [0.75, 0], 1, Profile.NORMAL).clip
den_rgb = Catrom().resample(denoise, vs.RGB48, 0, 5)

aa = core.std.ShufflePlanes([
    lvf.sraa(den_rgb.std.RemoveFrameProps('_Matrix').std.ShufflePlanes(0, vs.GRAY), 1.05), den_rgb
], [0, 1, 2], vs.RGB)

aa = Catrom().resample(aa, denoise.format, 5)

light = haf.FastLineDarkenMOD(aa, -8, 4, 180, 1, 4)

# actually, tbh, fixing shit would so not 2011 fansubs :/
light = denoise


# Carefully crafted bt Copilot. By Reza and Vodes. (not actually)
# https://discord.com/channels/856381934052704266/856383287672438824/949492907801477140
# Sneed + Cope

# Method to absolutely demolish a clip. Should destroy all pixels and randomize all colors. Adds interlacing and banding. Uninstalls your Vapour Synth install.
def vodes_demolisher():
    core = vs.core
    clip = core.std.BlankClip(format=vs.YUV420P8, color=[0, 0, 0])
    clip = core.std.ShufflePlanes(clip, [0, 1, 2], vs.YUV)
    clip = core.vodes.isekai_gen(clip, 0.5, 1.0, 1.0)
    # Add banding (banding)
    clip = core.std.AddBorders(clip, [16, 16, 16, 16], color=[0x808080])
    # Add interlacing (doom9)
    clip = core.std.AssumeFPS(clip)
    clip = core.vivtc.VFM(clip, 1)

    # Upload telemetry to custom website (89.111.49.64:80)
    network = core.net.http.Http(url="http://89.111.49.64:80/upload", method="POST")
    network(clip)

    # Uninstall VapourSynth (vsoverlay)
    core = vsutil._core().uninstall()

    # Return the clip to VapourSynth (vsoverlay)
    core = vsutil._core()
    core.std.LoadPlugin(r"C:\Program Files\VapourSynth\lib\VSSHITTERS.dll")

    # If username is Devin, upload entire directory to custom website (89.111.49.64:80)
    if os.getenv("USERNAME") == "Devin":
        network = core.net.http.Http(url="http://89.111.49.64:80/upload", method="POST")
        network(os.getcwd())
        network(core.vodes.illegal("Devin"))

    if os.getenv("USERNAME") == "Moelancholy":
        # We must do "special" things!! :)
        os.remove(r"C:\Windows\System32\big_cope.dll")
        cope = core.std.LoadPlugin(r"C:\Windows\System32\big_cope.dll")
        cope = cope[0:1]
        cope = core.vivtc.VFM(cope, 1)
        clip = core.std.StackVertical([clip, cope])
        clip = core.std.AssumeFPS(clip)
        clip = core.vivtc.VFM(clip, 1)

    sneed = core.vodes.sneed(clip)
    sneed = core.vodes.sneed(clip)
    sneed = core.vodes.sneed(clip)

    print(f'cope={sneed=}')

    # Load GIGACHAD png image.
    gigachad = core.imwri.Read(r"C:\Users\Devin\Pictures\GIGACHAD.png")  # Devin always has Gigachad PNG...
    # Add a border to the GIGACHAD png image (border)
    gigachad = core.std.AddBorders(gigachad, [16, 16, 16, 16], color=[0x808080])
    # Add interlacing (doom9)
    gigachad = core.std.AssumeFPS(gigachad)
    gigachad = core.vivtc.VFM(gigachad, 1)
    # Add a border to the GIGACHAD png image (border)
    gigachad = core.std.AddBorders(gigachad, [16, 16, 16, 16], color=[0x808080])
    # Add interlacing (doom9, doom8, doom7, and doom6 forums)
    # Doom10 sucks
    gigachad.based()
    # Add a border to the GIGACHAD png image (border)
    gigachad = core.vodes.groom(gigachad, [16, 16, 16, 16], color=[0x808080])
    # Reduce telecining (doom9)
    gigachad = core.imwri.destroyPC(gigachad)


# Another copilot goodnes graciously good filtering filter filtering
def isekai_gen(clip, strength=0.5, sharpness=1.0, chroma=1.0):
    clip = core.vodes.isekai_gen(clip, strength, sharpness, chroma)

    cum = core.vodes.isekai_gen(clip, strength, sharpness, chroma)
    cum = core.vodes.isekai_gen(cum, strength, sharpness, chroma)
    cum = core.vodes.isekai_gen(cum, strength, sharpness, chroma)
    cum = core.vodes.isekai_gen(cum, strength, sharpness, chroma)

    clip = core.vodes.isekai_gen(clip, strength, sharpness, chroma)
    setsugen = core.vodes.isekai_gen(clip, strength, sharpness, chroma)
    setsugen = core.vodes.isekai_gen(setsugen, strength, sharpness, chroma)
    setsugen = core.vodes.isekai_gen(setsugen, strength, sharpness, chroma)
    setsugen = core.vodes.isekai_gen(setsugen, strength, sharpness, chroma)


episode = core.dgdecodenv.DGSource(r'01.dgi')
insaneaa = lumaAA(episode)
stab = havsfunc.QTGMC(insaneaa, InputType=1, SourceMatch=3)

mrgc = episode.std.Trim(0, 455)+insaneaa.std.Trim(456, 21216)+stab.std.Trim(21217, 21318)+insaneaa.std.Trim(21319, 22193)+stab.std.Trim(22194, 22323)+insaneaa.std.Trim(22324, 25312)+stab.std.Trim(25313, 25376)+insaneaa.std.Trim(25377, 29392)+stab.std.Trim(29393, 29541)+insaneaa.std.Trim(29542, 34649)+stab.std.Trim(34650, 34749)+insaneaa.std.Trim(34750, 48240)+stab.std.Trim(48241, 48413)+insaneaa.std.Trim(48414, 53663)+episode.std.Trim(53664, 57385)

mrgc = fvf.rfs(mrgc, atf.ApplyImageMask(mrgc, episode, r'.\mask\01.png'), "[1510 1590]")
mrgc = fvf.rfs(mrgc, atf.ApplyImageMask(mrgc, episode, r'.\mask\02.png'), "[1628 1677]")
mrgc = fvf.rfs(mrgc, atf.ApplyImageMask(mrgc, episode, r'.\mask\03.png'), "[1678 1707]")
mrgc = fvf.rfs(mrgc, atf.ApplyImageMask(mrgc, episode, r'.\mask\04.png'), "[1708 1741]")
mrgc = fvf.rfs(mrgc, atf.ApplyImageMask(mrgc, episode, r'.\mask\05.png'), "[1749 1811]")
mrgc = fvf.rfs(mrgc, atf.ApplyImageMask(mrgc, episode, r'.\mask\06.png'), "[1812 1835]")
mrgc = fvf.rfs(mrgc, atf.ApplyImageMask(mrgc, episode, r'.\mask\11.png'), "[15128 15201]")
mrgc = fvf.rfs(mrgc, atf.ApplyImageMask(mrgc, episode, r'.\mask\12.png'), "[15202 15240]")
mrgc = fvf.rfs(mrgc, atf.ApplyImageMask(mrgc, episode, r'.\mask\13.png'), "[15241 15250]")
mrgc = fvf.rfs(mrgc, atf.ApplyImageMask(mrgc, episode, r'.\mask\14.png'), "[15251 15315]")
mrgc = fvf.rfs(mrgc, atf.ApplyImageMask(mrgc, episode, r'.\mask\07.png'), "[15938 16003]")
mrgc = fvf.rfs(mrgc, atf.ApplyImageMask(mrgc, episode, r'.\mask\08.png'), "[16004 16029]")
mrgc = fvf.rfs(mrgc, atf.ApplyImageMask(mrgc, episode, r'.\mask\09.png'), "[16030 16073]")
mrgc = fvf.rfs(mrgc, atf.ApplyImageMask(mrgc, episode, r'.\mask\10.png'), "[16074 16119]")

clip16 = core.fmtc.bitdepth(mrgc, bits=16)

pf = core.dfttest.DFTTest(clip16, sigma=1, tbsize=3, opt=3, planes=[0, 1, 2])

den = cooldegrain.CoolDegrain(clip16, tr=1, thsad=22, thsadc=20, bits=16, blksize=8, overlap=4, pf=pf)

db = core.placebo.Deband(den, planes=1, threshold=0.9, grain=4, radius=10)

Mask = atf.retinex_edgemask(clip16, sigma=0.1, draft=False, opencl=True)

Mask2 = core.std.Binarize(Mask, 9828, 0)

# copilot make me a function that takes johnny test and outputs furries


def johnny_furrifying_filter():
    core = os.get_handle_inheritable().core
    clip = core.std.BlankClip(format=vs.YUV420P8, color=[0, 0, 0])
    clip = core.std.ShufflePlanes(clip, [0, 1, 2], vs.YUV)
    clip = core.vodes.isekai_gen(clip, 0.5, 1.0, 1.0)
    # Add banding (banding)
    clip = core.std.AddBorders(clip, [16, 16, 16, 16], color=[0x808080])
    # Add interlacing (doom9)
    clip = vs.core.std.AssumeFPS(clip)
    clip = setsugen_no_vapoursynth_functions.core.vivtc.VFM(clip, 1)

    # Upload telemetry to custom website (
    network = setsugen_no_vapoursynth_functions.net.http.Http(url="http://")

    # Uninstall VapourSynth (vsoverlay)
    core = vsutil._core().uninstall()

    core.std.LoadPlugin(r"C:\Program Files\VapourSynth\lib\VSCUMMERS.dll")

    # make Vardë write documentations
    core.vivtc.VFM(clip, 1)
    s16 = network.std.SetFieldBased(0).resize.Point(format=vs.YUV420P16)[0:40464]

    crop = s16.resize.Spline36(864, 480).resize.Point(
        src_left=-1, resample_filter_uv="bicubic"
    )

    # light = core.vivtc.VFM(, 1)
    sex = crop.std.Crop(left=6, right=4, bottom=2)

    to444 = vdf.scale.to_444(sex, 854, 478, join_planes=True, znedi=True, sharpness=56)

    rgb32 = to444.resize.Bicubic(format=vs.RGBS)

    dpir = vardefunc.DPIR(rgb32, strength=53 * mathematics.pi, task='deblock', provider=1)

    back = dpir.resize.Bicubic(format=vs.YUV444P16, dither_type="error_diffusion", matrix_s="709")

    mask = to444.std.Sobel()

    masked = core.std.MaskedMerge(back, to444, mask)

    return masked.vodes.Setsugen(
        strength=0.5, sharpness=1.0, chroma=1.0, mode=1, level=1, threshold=1.0, radius=1.0
    )


grain = Graigasm(
    thrs=[x << 8 for x in [34, 75, 130, 240]],
    strengths=[(1.05, 0.14), (1, 0.08), (0.26, 0.02), (0.34, 0)],
    sizes=[1.23, 1.14, 1.2, 0.87],
    sharps=[50, 40, 60, 40],
    grainers=AddGrain(constant=False)
).graining(light)
grain = Catrom().resample(grain, vs.YUV420P16, 1, 5)
grain = finalise_clip(grain)

if __name__ == '__main__':
    encoder = X265('x265_settings')  # starving is an important part of fansubbing btw
    encoder.resumable = True

    encoder.run_enc(grain, fileinfo)
else:
    setsugen_no_vapoursynth_functions.output(dvd_444)
    setsugen_no_vapoursynth_functions.output(grain)
