# Draft that uses motion analysis stuff to clean __more better__ and not wipe details
# debanding isn't as strong and i don't feel like fixing this as it's not going to be run lol
from ccd import ccd
import stgfunc as stg
import lvsfunc as lvf
import havsfunc as haf
import vapoursynth as vs
from vsmlrt import Backend
from mvsfunc import LimitFilter
from vsmask.edge import PrewittStd
from lvsfunc.deblock import vsdpir
from rgvs import repair, removegrain
from stgfunc.deband import auto_deband
from stgfunc.tweaking import bbmod_fast
from EoEfunc.frequencies import SLocation
from vardefunc.aa import Nnedi3SS, Eedi3SR
from lvsfunc.kernels import Catrom, RobidouxSharp
from EoEfunc.denoise import CMDegrain, GlobalMode
from vardefunc.misc import YUVPlanes, merge_chroma
from vsmask.util import expand, inpand, XxpandMode
from fine_dehalo import contrasharpening_fine_dehalo
from vsdenoise import BM3DCudaRTC, knl_means_cl, Profile
from vardefunc.util import initialise_clip, finalise_clip
from debandshit import f3kpf, f3kbilateral, placebo_deband
from vardautomation import FileInfo, PresetBD, X265, get_vs_core
from vsutil import get_neutral_value, get_peak_value, get_y, split, join


core = get_vs_core(range(0, vs.core.num_threads, 2))
catrom = Catrom()
robidx = RobidouxSharp()

JPBD = FileInfo(
    r"E:\Desktop\Encoding Sources\BDMVs - ISOs\Summer Ghost\SUMMER_GHOST\BDMV\STREAM\00000.m2ts",
    (456, -24), preset=[PresetBD]
)

src = initialise_clip(JPBD.clip_cut)

neutral = get_neutral_value(src)
peak = get_peak_value(src)

src = bbmod_fast(src, 1, 1)

fuck_compression = core.average.Mean([
    src.deblock.Deblock(quant) for quant in range(16, 32, 4)
])

sloc_low = SLocation([0.0, 4, 0.15, 512, 0.45, 16, 0.55, 512, 1.0, 512]).interpolated('cubic').to_list()
sloc_mid = SLocation([0, 1.5, 0.45, 2.5, 0.35, 2.45, 0.8, 2.14, 0.53, 0.97]).interpolated('quadratic').to_list()
sloc_high = SLocation([0.0, 512, 0.15, 4, 0.35, 16, 0.4, 512, 1.0, 512]).interpolated('quadratic').to_list()

denoisedft = src.dfttest.DFTTest(tbsize=1, slocation=sloc_mid)
denoisebm3d = BM3DCudaRTC(src, [2.65, 0.45], 1, Profile.HIGH).clip
den_rgb = catrom.resample(denoisebm3d, vs.RGBS, 0, 1)

degrain_args = dict(tr=1, thSAD=115, thSADC=60)

CMDegrain(fuck_compression, globalmode=GlobalMode.WRITE, **degrain_args)

curippusu = [
    denoisebm3d, fuck_compression.rgvs.RemoveGrain(16), denoisedft,
    CMDegrain(denoisedft, globalmode=GlobalMode.READ, **degrain_args),
    CMDegrain(denoisebm3d, globalmode=GlobalMode.READ, **degrain_args)
]

average = core.average.Mean(curippusu)
median = core.median.Median(curippusu)

merge = denoisedft

low_freq = merge.dfttest.DFTTest(slocation=sloc_high)
high_freq = average.dfttest.DFTTest(slocation=sloc_high)
merge = low_freq.std.MergeDiff(average.std.MakeDiff(high_freq))
low_freq = merge.dfttest.DFTTest(slocation=sloc_low)
high_freq = denoisebm3d.dfttest.DFTTest(slocation=sloc_low)
merge = low_freq.std.MergeDiff(denoisebm3d.std.MakeDiff(high_freq))

contra_rgb = Catrom().resample(merge, vs.RGBS, 0, 1)
cdenoise = ccd(contra_rgb, 15)
cdenoise = core.akarin.Expr([cdenoise, contra_rgb, den_rgb], ['x y min z min', 'x z + 2 / y min', 'x y z + + 3 /'])
# cdenoise = core.expr.expr_cpp(   # <--- Wasn't that much faster than akarin :sadge:
#     [cdenoise, contra_rgb, den_rgb], ['cdenoise_R', 'cdenoise_G', 'cdenoise_B'], source_path='sghost_expr.cpp'
# )
cdenoise = catrom.resample(cdenoise, vs.YUV420P16, 1)

denoise = knl_means_cl(cdenoise, [0.16, 0.25], rclip=merge)

denoise = vsdpir(denoise, 5, 'deblock', 1, backend=Backend.TRT)

denoise = merge_chroma(denoise, denoise)
denoise = contrasharpening_fine_dehalo(denoise, median, 3)

with YUVPlanes(denoise) as c:
    deb_args = dict(
        downsample_h=720, topk=0.005, tvi_threshold=0.002,
        debug=(False, False), ref_clip=get_y(src), cambi_scale=3
    )

    adptvgr_args = dict(
        luma_scaling=8.5, static=False, size=0.95, sharp=45
    )

    y = c.Y

    light_deband = auto_deband(
        y, cambi_thr=1, min_thr=18, max_thr=32,
        steps=4, grain_thrs=(0.1, 0.13, 0.16),
        debander=f3kpf, debander_args=dict(
            limflt_args=dict(elast=4), radius=12
        ), adptvgr_args=adptvgr_args, **deb_args
    )

    hard_deband = auto_deband(
        light_deband, min_thr=48, max_thr=64,
        steps=6, grain_thrs=(0.18, 0.26, 0.2),
        debander=f3kbilateral, debander_args=dict(
            limflt_args=dict(thr=0.3), radius=15
        ), adptvgr_args=adptvgr_args, **deb_args
    )

    stronger_deband = auto_deband(
        hard_deband, min_thr=8, max_thr=12,
        steps=4, grain_thrs=(0.2, 0.3, 0.14),
        debander=placebo_deband, debander_args=dict(
            iterations=8, radius=16
        ), adptvgr_args=adptvgr_args, **deb_args
    )

    c.Y = stronger_deband

deband = c.clip

smoothed = haf.MinBlur(deband, 2)

repclp = core.rgvs.Repair(deband, smoothed, mode=[24, 0, 0])

limitclp = LimitFilter(repclp, deband, thr=12, elast=2, brighten_thr=3)

prewitt = PrewittStd().edgemask(catrom.resample(deband, vs.RGB48, 0, 1)).std.Expr(f'x {60 << 16} < 0 x ?')

prewittm = core.std.Expr(split(prewitt), 'x y z max max').std.RemoveFrameProps('_Matrix').std.Minimum().std.Inflate()
fmask = core.misc.Hysteresis(prewittm.std.Median(), prewittm)
omask = inpand(
    expand(fmask, 1, 1, mode=XxpandMode.ELLIPSE), 1, 1, mode=XxpandMode.ELLIPSE
)

imask = fmask.std.Inflate().std.Minimum()

ringmask = core.std.Expr([omask, imask], f'x {peak} y - * {peak} /')

dering = deband.std.MaskedMerge(limitclp, ringmask)

dering_y = get_y(dering)

fsrcnnx = get_y(core.placebo.Shader(
    join([dering_y] * 3), stg.misc.x56_SHADERS, dering.width * 2, dering.height * 2, filter='box'
))

smooth = Nnedi3SS(False, 0).scale(dering_y, dering.width * 2, dering.height * 2)

halo_mask = lvf.mask.halo_mask(smooth)

upscale = core.akarin.Expr(
    [fsrcnnx, smooth, smooth.std.Maximum(), smooth.std.Minimum(), halo_mask], f"""
    x 0.8 * y 0.2 * + up! {1.5 * 2**8} O! {1.1 * 2**8} U! up@ z O@ + > z O@
    + up@ ? a U@ - < a U@ - up@ z O@ + > z O@ + up@ ? ? 16 255 * 235 255 * clamp UP!
    x y min MIN! b {peak} / UP@ * 1 b {peak} / - MIN@ * +
""")

mask = PrewittStd().edgemask(upscale).akarin.Expr(
    f'x 4 255 * < 0 x 32 255 * > {peak} x ? ?'
).std.InvertMask().std.Convolution([1] * 9)

final = repair(upscale, removegrain(upscale, 11), 2)

diff = upscale.std.MakeDiff(removegrain(upscale, 15))

mask = PrewittStd().edgemask(diff.std.Levels(40 << 8, 168 << 8, 0.35))

mask = mask.akarin.Expr(f"""
    x[-1,-1] x[1,1] min mi1! x[-1,-1] x[1,1] max ma1! x[0,-1] x[0,1] min mi2! x[0,-1] x[0,1] max ma2! x[1,-1]
    x[-1,1] min mi3! x[1,-1] x[-1,1] max ma3! x[-1,0] x[1,0] min mi4! x[-1,0] x[1,0] max ma4!
    x mi1@ ma1@ clamp clamp1! x mi2@ ma2@ clamp clamp2! x mi3@ ma3@ clamp clamp3! x mi4@ ma4@ clamp clamp4!
    x clamp1@ - abs ma1@ mi1@ - + c1! x clamp2@ - abs ma2@ mi2@ - + c2! x clamp3@ - abs ma3@ mi3@ - +
    c3! x clamp4@ - abs ma4@ mi4@ - + c4! c1@ c2@ c3@ c4@ min min min mindiff!
    mindiff@ c4@ = clamp4@ mindiff@ c2@ = clamp2@ mindiff@ c3@ = clamp3@ clamp1@ ? ? ? m!
    m@ 4 255 * < 0 m@ 16 255 * > {peak} m@ ? ?
""")

repaired = final.std.MaskedMerge(upscale, mask)

singlerate = Eedi3SR(
    False, False, 0.15, 0.55, 420, nrad=3, mdis=12,
    eedi3_args=dict(vcheck=3), nnedi3_args=dict(nsize=0)
).aa(repaired.std.Transpose()).std.Transpose()

aa = merge_chroma(catrom.scale(singlerate, src.width, src.height), dering)

neutral_clip = src.std.BlankClip(color=[neutral] * 3)

heavy_grain = aa.grain.Add(1.25, 0.43)

original_grain_diff = denoise.std.MakeDiff(src).std.MaskedMerge(neutral_clip, prewittm)

rg11 = core.akarin.Expr([
    original_grain_diff.akarin.Expr(f'{neutral} n! x x 2 * x[0,-1] x[0,1] + + 4 / - n@ +'), original_grain_diff
], f"""
    {neutral} n! x 2 * x[0,-1] x[0,1] + + 4 / blur! y x blur@ - x n@ - * 0 < n@ x blur@
    - abs x n@ - abs < x blur@ - n@ + x ? ? - n@ +
""")

original_grain_diff = core.akarin.Expr([original_grain_diff, rg11], f"""
    {neutral} n! 4 255 * a! y y y y 2 * y[0,-1] y[0,1] + + 4 / - 2.7 * + y - n@ + sdiff! x
    y - n@ + diff! sdiff@ n@ - diff@ n@ - * 0 < sdiff@ n@ - abs diff@ n@ - abs < sdiff@ diff@ ?
    n@ - 0.65 * n@ + sdiff@ n@ - abs diff@ n@ - abs < sdiff@ diff@ ? ? n@ - + merge! x a@ +
    merge@ < x a@ + x a@ - merge@ > x a@ - merge@ ? ?
""")

grain = heavy_grain.std.MergeDiff(original_grain_diff)
grain = finalise_clip(grain)

if __name__ == "__main__":
    encoder = X265('x265_settings')
    encoder.resumable = True

    encoder.run_enc(grain, JPBD, qpfile_clip=catrom.scale(src, 1280, 720))
else:
    stg.output(src)
    stg.output(grain)
    stg.output(prewittm)
