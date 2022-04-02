from ccd import ccd
import mvsfunc as mvf
import stgfunc as stg
import lvsfunc as lvf
import havsfunc as haf
import vapoursynth as vs
from mvsfunc import LimitFilter
from rgvs import repair, removegrain
from stgfunc.tweaking import bbmod_fast
from EoEfunc.frequencies import SLocation
from vsdenoise import BM3DCuda, Profile
from vsmask.util import expand, XxpandMode
from vsmask.edge import PrewittStd, FDoGTCanny
from vardefunc.noise import Graigasm, AddGrain
from lvsfunc.kernels import Catrom, RobidouxSharp
from vardefunc.misc import YUVPlanes, merge_chroma
from debandshit import f3kbilateral, placebo_deband
from fine_dehalo import contrasharpening_fine_dehalo
from vardefunc.aa import Nnedi3SS, Eedi3SR, SangNomSR
from vardefunc.util import initialise_clip, finalise_clip
from vardautomation import FileInfo, PresetBD, X265, get_vs_core  # , Patch, VPath
from vsutil import get_neutral_value, get_peak_value, get_y, split, join, iterate, depth


core = get_vs_core(range(0, vs.core.num_threads, 2))
catrom = Catrom()
robidx = RobidouxSharp()

JPBD = FileInfo(
    r"E:\Desktop\Encoding Sources\BDMVs - ISOs\Summer Ghost\SUMMER_GHOST\BDMV\STREAM\00000.m2ts",
    (456, -24), preset=[PresetBD]
)

src = initialise_clip(JPBD.clip_cut, 8)

src = bbmod_fast(src, 1, 1)

sloc_mid = SLocation([0, 1.5, 0.45, 2.5, 0.35, 2.45, 0.8, 2.14, 0.53, 0.97]).interpolated('quadratic').to_list()

denoisedft = src.dfttest.DFTTest(tbsize=1, slocation=sloc_mid)
denoisebm3d = BM3DCuda(src, [2.65, 0.45], 1, Profile.FAST).clip

contra_rgb = catrom.resample(denoisedft, vs.RGBS, 0, 1)
cdenoise = ccd(contra_rgb, 15)
cdenoise = core.akarin.Expr([
    cdenoise, contra_rgb, catrom.resample(denoisebm3d, vs.RGBS, 0, 1)
], [
    'x y min z min', 'x z + 2 / y min', 'x y z + + 3 /'
])
denoise = catrom.resample(cdenoise, vs.YUV420P16, 1)
src, denoisebm3d = depth(src, 16), depth(denoisebm3d, 16)

neutral = get_neutral_value(src)
peak = get_peak_value(src)

with YUVPlanes(denoise) as c:
    y = c.Y

    hard_deband = f3kbilateral(y, 8, 48, 0, None, dict(thr=0.3))

    stronger_deband = placebo_deband(hard_deband, 16, 6, 6, 2.4)

    c.Y = lvf.rfs(stronger_deband, f3kbilateral(
        placebo_deband(stronger_deband, 24, 8, 10, 0), threshold=48,
        limflt_args=dict(thr=1, elast=3.65)
    ), [(14451, 14582), (17779, 17976)])

deband = c.clip

smoothed = haf.MinBlur(deband, 2)

repclp = core.rgvs.Repair(deband, smoothed, mode=[24, 0, 0])

limitclp = LimitFilter(repclp, deband, thr=12, elast=2, brighten_thr=3)

prewitt = PrewittStd().edgemask(catrom.resample(deband, vs.RGB48, 0, 1)).std.Expr(f'x {35 << 8} < 0 x ?')

prewittm = core.std.Expr(split(prewitt), 'x y z max max').std.RemoveFrameProps('_Matrix').std.Minimum().std.Inflate()
fmask = core.misc.Hysteresis(prewittm.std.Median(), prewittm)
omask = expand(fmask, 1, 1, mode=XxpandMode.ELLIPSE)

ringmask = core.std.Expr([omask, fmask.std.Inflate().std.Minimum()], f'x {peak} y - * {peak} /')

dering = deband.std.MaskedMerge(limitclp, ringmask)

dering_y = get_y(dering)

fsrcnnx = get_y(core.placebo.Shader(
    join([dering_y] * 3), stg.misc.x56_SHADERS, dering.width * 2, dering.height * 2, filter='box'
))

smooth = Nnedi3SS(False, 0).scale(dering_y, dering.width * 2, dering.height * 2)

halo_mask = lvf.mask.halo_mask(smooth)

upscale = core.akarin.Expr(
    [fsrcnnx, smooth, smooth.std.Maximum(), fsrcnnx.std.Minimum(), halo_mask], f"""
    x 0.8 * y 0.2 * + up! {1.5 * 2**8} O! {1.1 * 2**8} U! up@ z O@ + > z O@
    + up@ ? a U@ - < a U@ - up@ z O@ + > z O@ + up@ ? ? 16 255 * 235 255 * clamp UP!
    x y max MAX! b {peak} / UP@ * 1 b {peak} / - MAX@ * +
""")

mask = PrewittStd().edgemask(upscale).akarin.Expr(
    f'x 4 255 * < 0 x 32 255 * > {peak} x ? ?'
).std.InvertMask().std.Convolution([1] * 9)

final = repair(upscale, removegrain(upscale, 11), 2)

diff = upscale.std.MakeDiff(removegrain(upscale, 15))

mask = PrewittStd().edgemask(diff.std.Levels(40 << 8, 168 << 8, 0.35))

repaired = final.std.MaskedMerge(upscale, mask)

eedisr = Eedi3SR(
    True, False, 0.15, 0.55, 380, nrad=3, mdis=12, nnedi3_args=dict(nsize=0)
).aa(repaired.std.Transpose()).std.Transpose()

aa = merge_chroma(catrom.scale(eedisr, src.width, src.height), dering)
aa = contrasharpening_fine_dehalo(aa, denoisebm3d, 1.4)

dark_edges = FDoGTCanny().edgemask(catrom.resample(aa, vs.RGB48), 0)
dark_edges = core.std.Expr(split(dark_edges), 'y z + 2 /').std.BinarizeMask(60 << 8).std.InvertMask().std.Inflate()
dark_edges = dark_edges.std.Maximum().std.Minimum().std.Inflate().std.Deflate()

eedisr1 = Eedi3SR(
    True, False, 0.2, 0.8, 20
).aa(SangNomSR(12).aa(aa)).std.MaskedMerge(aa, dark_edges)

aa = lvf.rfs(aa, eedisr1, (7310, 7312))  # i am sorry for those lines at the right

pref = iterate(get_y(aa), core.std.Maximum, 2).std.Convolution([1] * 9)

grain = Graigasm(
    thrs=[x << 8 for x in (58, 80, 128)],
    strengths=[(0.94, 0.285), (0.45, 0.18), (0.24, 0.15)],
    sizes=(1.56, 1.46, 1.5),
    sharps=(55, 45, 40),
    grainers=[
        AddGrain(seed=80085, constant=False),
        AddGrain(seed=69420, constant=True),
        AddGrain(seed=69420, constant=True)
    ]
).graining(aa, prefilter=pref).std.MergeDiff(
    src.std.MakeDiff(denoise).std.MaskedMerge(
        src.std.BlankClip(color=[neutral] * 3), fmask.resize.Bilinear(1920, 1080).std.Inflate()
    )
)

# creditss = src[53396:53396+3275]

# creditss_pan = core.std.Interleave([
#     creditss,
#     creditss.resize.Bicubic(src_top=1.3),
#     creditss.resize.Bicubic(src_top=2.6)
# ]).std.AssumeFPS(None, 24000 * 3, 1001)

# grain = mvf.VFRSplice([
#     grain[:53208] + src[53208:53396], creditss_pan, grain[53396+3275:]
# ], './timecodes.txt' if __name__ == "__main__" else None, True, 6)

grain = grain[:53208] + src[53208:]

grain = finalise_clip(grain)

if __name__ == "__main__":
    encoder = X265('x265_settings')
    # encoder = X265('x265_settings_credits')
    encoder.resumable = True

    grain = grain.std.AssumeFPS(None, 24000, 1001)

    # JPBD.name_file_final = VPath(r'.\Premuxes\Summer Ghost v0 [premux].mkv')
    # patch = Patch(encoder, grain, JPBD, [
    #     (7310, 7312+1), (17779, 17976+1)
    # ])

    # patch.run()
    # patch.do_cleanup()

    # encoder.run_enc(grain[53208:], JPBD)
    encoder.run_enc(grain, JPBD, qpfile_clip=catrom.scale(src, 1280, 720))
else:
    stg.output(src)
    stg.output(grain)
    stg.output(dark_edges.std.RemoveFrameProps('_Matrix'))
