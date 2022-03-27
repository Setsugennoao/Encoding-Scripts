from ccd import ccd
import stgfunc as stg
import lvsfunc as lvf
import havsfunc as haf
import vapoursynth as vs
from vsdenoise import BM3DCuda
from vardefunc.aa import Eedi3SR
from vardefunc.misc import merge_chroma
from stgfunc.deband import masked_f3kdb
from vardefunc.noise import Graigasm, AddGrain
from lvsfunc.kernels import Catrom, BicubicDidee
from vardefunc.scale import fsrcnnx_upscale, to_444
from vsutil import depth, get_y, join, split, plane
from fine_dehalo import fine_dehalo, contrasharpening
from vardefunc.util import initialise_clip, finalise_clip
from vardautomation import FileInfo, PresetWEB, X265, get_vs_core

from jojo_so_commons.scaling import Scaling

core = get_vs_core(range(0, vs.core.num_threads, 2))
catrom = Catrom()
no_rescale_ranges = []

fileinfov0 = FileInfo(
    r"E:\Desktop\Encoding Sources\JoJo Stone Ocean\JoJo Stone Ocean - EP04v0 (01.12.21).mkv",
    (323, 34230), idx=stg.src, preset=[PresetWEB]
)

fileinfov1 = FileInfo(
    r"E:\Desktop\Encoding Sources\JoJo Stone Ocean\JoJo Stone Ocean - EP04v1 (16.01.22).mkv",
    (323, 34230), idx=stg.src, preset=[PresetWEB]
)

srcv0 = initialise_clip(fileinfov0.clip_cut).resize.Bicubic(chromaloc_in=1, chromaloc=0)
srcv1 = initialise_clip(fileinfov1.clip_cut).resize.Bicubic(chromaloc_in=1, chromaloc=0)

srcv0 = lvf.rfs(srcv0, srcv1, [
    (108, 171), (4886, 4903), (10505, 10573),
    (15998, 16009), (16438, 16488), (20031, 20320),
    (20502, 20838), (26417, 26716)
])

# stg.output(lvf.diff(srcv0, srcv1))

oycore = stg.oysterV2.Oyster()

median = core.median.Median([srcv0, oycore.FreqMerge(srcv1, srcv0), srcv1])

y = get_y(median)

y_den = BM3DCuda(y, 0.35, 1).clip
y_den = contrasharpening(y, get_y(srcv0))

dehalo = fine_dehalo(y_den, None, 2, 2, 0, 1.15, contra=1)

descaled = Scaling.descale(dehalo)
descaled = depth(descaled, 16)

upscale = fsrcnnx_upscale(descaled, shader_file=stg.misc.x56_SHADERS, downscaler=None)

eedaa = Eedi3SR(alpha=0.2, beta=0.5, gamma=485, nrad=3, mdis=15).do_aa()(upscale)

rescaled = catrom.scale(eedaa, y.width, y.height)

upscale = merge_chroma(rescaled, median)

recon = lvf.recon.chroma_reconstruct(
    upscale, 1, True
).warp.AWarpSharp2(
    thresh=92, blur=3, type=1, depth=6, planes=[1, 2]
).warp.AWarpSharp2(
    thresh=68, blur=2, type=1, depth=-2, planes=[0]
).warp.AWarpSharp2(
    thresh=68, blur=2, type=1, depth=3, planes=[0]
)

recon = lvf.rfs(recon, to_444(merge_chroma(y_den, median), None, None, True, True), (31759, None))

recon = ccd(ccd(ccd(recon, 2), 4), 6)
recon = BicubicDidee().resample(recon, vs.YUV420P16)

recon = join([contrasharpening(rconp, upscp) for rconp, upscp in zip(split(recon), split(upscale))])

edgeclean = haf.EdgeCleaner(recon, 7, True, smode=1, hot=True)

deblock2 = lvf.deblock.vsdpir(edgeclean, 25, tile_x=960, tile_y=540, tile_pad=16, trt=True, fp16=True)

deblock = join([
    contrasharpening(get_y(deblock2), get_y(edgeclean)),
    core.average.Mean([plane(edgeclean, 1), plane(deblock2, 1), plane(recon, 1)]),
    core.average.Mean([plane(edgeclean, 2), plane(deblock2, 2), plane(recon, 2)])
])

deband = masked_f3kdb(deblock)

grain = Graigasm(
    thrs=[x << 8 for x in (40, 90, 168)],
    strengths=[(0.74, 0.15), (0.35, 0.1), (0.25, 0.06)],
    sizes=(1.2, 1.16, 1.25),
    sharps=(55, 45, 40),
    grainers=[
        AddGrain(seed=80085, constant=False),
        AddGrain(seed=69420, constant=True),
        AddGrain(seed=69420, constant=True)
    ]
).graining(deband)

grain = finalise_clip(grain)

if __name__ == '__main__':
    encoder = X265('jojo_so_commons/x265_settings')

    fileinfov0.name_clip_output = fileinfov0.workdir / 'jojo_so_04v1.265'

    encoder.run_enc(grain, fileinfov0)
else:
    src = srcv1
    stg.output(src)
    stg.output(grain)
