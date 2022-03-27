import stgfunc as stg
import lvsfunc as lvf
import havsfunc as haf
from vsmask.edge import FDOG
from functools import partial
from vsutil import Dither, get_w
from vardautomation import get_vs_core
from vardefunc.misc import merge_chroma
from stgfunc.deband import masked_f3kdb
from vardefunc.util import finalise_clip
from vardautomation import X265, FileInfo
from lvsfunc.scale import ssim_downsample
from vsdenoise.bm3d import BM3DCuda, Profile
from lvsfunc.kernels import Catrom, Spline64
from vardefunc.noise import Graigasm, AddGrain
from vardefunc.aa import Eedi3SR, upscaled_sraa
from debandshit import dumb3kdb, placebo_deband
from vardefunc.scale import to_444, fsrcnnx_upscale
from vsdenoise.knlm import knl_means_cl, ChannelMode
from fine_dehalo import fine_dehalo, contrasharpening
from vsutil import get_y, depth, iterate, plane, join

core = get_vs_core(range(0, 24, 2))
catrom = Catrom()

fileinfo = FileInfo(r".\Source\Sono Bisque Doll wa Koi wo Suru - 02 - (CR CBR 1080p).mkv",
                    idx=partial(stg.src, depth=16, matrix_prop=1))
crunchy = fileinfo.clip_cut

op_ranges, ed_ranges = (600, 2758), (31771, 33928)
no_rescale = [(op_ranges[1] + 1, op_ranges[1] + 72), ed_ranges, (ed_ranges[1] + 1, None)]

cden = knl_means_cl(crunchy, 1.05, channels=ChannelMode.CHROMA)

recon = lvf.recon.chroma_reconstruct(cden, 1, True).warp.AWarpSharp2(
    thresh=92, blur=3, type=1, depth=6, planes=[1, 2]
)

recon = join([
    plane(recon, 0),
    plane(recon, 1).bilateralgpu.Bilateral(0.26),
    plane(recon, 2).bilateralgpu.Bilateral(0.65)
])

recon = lvf.rfs(recon, to_444(cden, None, None, True), ed_ranges)

denoise = BM3DCuda(recon, [1.3, 0.36], 1, Profile.NORMAL).clip

y = get_y(denoise)

y_32 = depth(y, 32, dither_type=Dither.NONE)

lineart_mask = FDOG().edgemask(y_32.rgsf.RemoveGrain(3).rgsf.RemoveGrain(3))

descale = Spline64().descale(y_32, get_w(872), 872)
descale = fine_dehalo(descale, rx=2.1, ry=2.1, darkstr=0, brightstr=0.85)
upscale = fsrcnnx_upscale(descale, shader_file=stg.misc.x56_SHADERS, downscaler=None, profile='zastin')
upscale_op = fsrcnnx_upscale(y_32, upscale.width, upscale.height, stg.misc.x56_SHADERS, ssim_downsample, strength=60)
upscale = lvf.rfs(upscale, upscale_op, op_ranges)

eedaa = Eedi3SR(eedi3cl=True, alpha=0.2, beta=0.5, gamma=500, nrad=3, mdis=15).do_aa()(upscale)

up_sraa = upscaled_sraa(descale, 3, eedaa.width, eedaa.height)
up_sraa_op = upscaled_sraa(y_32, 2, eedaa.width, eedaa.height)
up_sraa = lvf.rfs(up_sraa, up_sraa_op, op_ranges)

aaa = lvf.aa.clamp_aa(upscale, eedaa, up_sraa, 0.7)

ssim = ssim_downsample(aaa, 1920, 1080)
catr = catrom.scale(aaa, 1920, 1080)

rescale = ssim.std.MaskedMerge(ssim.std.Merge(catr, 2 / 7), lineart_mask)
rescale = depth(rescale, 16)
rescale = contrasharpening(rescale, y, 2).std.MaskedMerge(rescale, depth(lineart_mask.std.Maximum(), 16))

dehalo = fine_dehalo(rescale, rx=2, ry=2, darkstr=0, brightstr=0.85)
dehalo = merge_chroma(dehalo, denoise)
dehalo = lvf.rfs(dehalo, denoise, no_rescale)

deband1 = dumb3kdb(dehalo)

deband2 = core.average.Mean([
    deband1,
    haf.FastLineDarkenMOD(dehalo, 30, 3, 190, 4, 1),
    dumb3kdb(deband1, 8, 18),
    dumb3kdb(dehalo, 12, 24),
    dumb3kdb(dehalo, 18, 30),
    dumb3kdb(dehalo, 24, 34),
    placebo_deband(deband1, 8, grain=2),
    placebo_deband(dehalo, 12, grain=3),
    placebo_deband(dehalo, 16, grain=5)
]).std.MaskedMerge(dehalo, stg.mask.linemask(y))

deband3 = stg.oyster.Core().FreqMerge(deband2, dehalo)

deband = masked_f3kdb(contrasharpening(masked_f3kdb(dehalo), deband3, 2), )

pref = iterate(get_y(deband), core.std.Maximum, 2).std.Convolution([1] * 9)

grain = Graigasm(
    thrs=[x << 8 for x in (40, 90, 138)],
    strengths=[(0.74, 0.15), (0.35, 0.1), (0.15, 0.06)],
    sizes=(0.8, 0.76, 0.65),
    sharps=(55, 45, 40),
    grainers=[
        AddGrain(seed=80085, constant=False),
        AddGrain(seed=69420, constant=True),
        AddGrain(seed=69420, constant=True)
    ]
).graining(deband, prefilter=pref)

grain = finalise_clip(grain)

if __name__ == '__main__':
    encoder = X265('bisque_commons/x265_settings')

    fileinfo.name_clip_output = fileinfo.workdir / 'bisque_doll_02v0.265'

    encoder.run_enc(grain, fileinfo)
else:
    stg.output(crunchy, False)
    stg.output(grain, False)
