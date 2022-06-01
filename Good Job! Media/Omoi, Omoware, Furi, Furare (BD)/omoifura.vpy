from ccd import ccd
import stgfunc as stg
import lvsfunc as lvf
import vapoursynth as vs
from pathlib import Path
from stgfunc.deband import auto_deband
from vsdenoise.knlm import knl_means_cl
from debandshit import f3kpf, f3kbilateral
from vsdenoise.bm3d import BM3DCuda, Profile
from vardefunc.noise import Graigasm, AddGrain
from vardefunc.misc import YUVPlanes, merge_chroma
from fine_dehalo import contrasharpening_fine_dehalo
from vardefunc.util import initialise_clip, finalise_clip
from vsutil import get_neutral_value, get_peak_value, plane
from vardautomation import FileInfo, PresetBD, get_vs_core, X265
from lvsfunc.kernels import BlackManMinLobe, RobidouxSoft, Catrom


core = get_vs_core(range(0, vs.core.num_threads, 2))

catrom = Catrom()
robisoft = RobidouxSoft()
pocminlobe = BlackManMinLobe()


JPBD = FileInfo(
    r"E:\Desktop\Encoding Sources\BDMVs - ISOs\[BDMV][アニメ]思い、思われ、ふり、ふられ (初回生産限定特別版)\BD\BDMV\STREAM\00001.m2ts",
    (456, 147851), preset=[PresetBD]
)

src = initialise_clip(JPBD.clip_cut, 16)

neutral = get_neutral_value(src)
peak = get_peak_value(src)

rg11 = core.akarin.Expr([
    src.akarin.Expr(f'{neutral} n! x x 2 * x[0,-1] x[0,1] + + 4 / - n@ +'), src
], f"""
    {neutral} n! x 2 * x[0,-1] x[0,1] + + 4 / blur! y x blur@ - x n@ - * 0 < n@ x blur@
    - abs x n@ - abs < x blur@ - n@ + x ? ? - n@ +
""")

fuck_m2ts = core.akarin.Expr([src, rg11], f"""
    {neutral} n! 4 255 * a! y y y y 2 * y[0,-1] y[0,1] + + 4 / - 2.7 * + y - n@ + sdiff! x
    y - n@ + diff! sdiff@ n@ - diff@ n@ - * 0 < sdiff@ n@ - abs diff@ n@ - abs < sdiff@ diff@ ?
    n@ - 0.65 * n@ + sdiff@ n@ - abs diff@ n@ - abs < sdiff@ diff@ ? ? n@ - + merge! x a@ +
    merge@ < x a@ + x a@ - merge@ > x a@ - merge@ ? ?
""")

m2y = plane(fuck_m2ts, 0)

den = BM3DCuda(m2y, 6, 1, Profile.FAST).clip
bidh = den.bilateral.Bilateral(
    den.bilateral.Bilateral(None, 1.25, 4.5 / 255), 1.4 / 3, 6 / 255
)

contra = contrasharpening_fine_dehalo(bidh, m2y, 2)

clean = stg.oyster.Core().FreqMerge(
    core.std.Expr([contra, m2y], 'x y max'), contra
)

denoise = BM3DCuda(clean, 1.46, 1, Profile.LOW_COMPLEXITY, refine=2).clip

denoise = ccd(merge_chroma(denoise, fuck_m2ts), 4.5)

rclip = pocminlobe.scale(plane(denoise, 0), 960, 540)

denoise = core.std.ShufflePlanes([
    plane(denoise, 0),
    knl_means_cl(plane(denoise, 1), 1.0, 0, 2, 2, rclip=rclip),
    knl_means_cl(plane(denoise, 2), 1.0, 0, 2, 2, rclip=rclip)
], [0, 0, 0], vs.YUV).std.Limiter()

clean = lvf.aa.based_aa(denoise, stg.misc.x56_SHADERS)

with YUVPlanes(clean) as c:
    deb_args = dict(
        downsample_h=720, topk=0.005, tvi_threshold=0.0025, ref_clip=plane(clean, 0), cambi_scale=5
    )

    adptvgr_args = dict(
        luma_scaling=8.5, static=False, size=0.95, sharp=45
    )

    light_deband = auto_deband(
        c.Y, cambi_thr=1, min_thr=18, max_thr=32,
        steps=4, grain_thrs=(0.1, 0.13, 0.16),
        debander=f3kpf, debander_args=dict(
            limflt_args=dict(elast=2), radius=8
        ), adptvgr_args=adptvgr_args, **deb_args
    )

    hard_deband = auto_deband(
        light_deband, min_thr=48, max_thr=64,
        steps=6, grain_thrs=(0.18, 0.26, 0.2),
        debander=f3kbilateral, debander_args=dict(
            limflt_args=dict(thr=0.3), radius=16
        ), adptvgr_args=adptvgr_args, **deb_args
    )

    c.Y = hard_deband

deband = c.clip

hardsub = None
for file in sorted(Path('./hardsub').glob('*.png')):
    frame = stg.src(file)
    hardsub = (hardsub + frame) if hardsub is not None else frame

deband = deband[:137370] + catrom.resample(
    hardsub, deband.format, 1
).std.AssumeFPS(deband) + deband[137370 + hardsub.num_frames:]

grain = Graigasm(
    thrs=[x << 8 for x in (40, 90, 168)],
    strengths=[(0.85, 0.15), (0.55, 0.1), (0.4, 0.03)],
    sizes=(1.16, 0.95, 0.8),
    sharps=(55, 45, 50),
    grainers=[
        AddGrain(seed=80085, constant=False),
        AddGrain(seed=69420, constant=False),
        AddGrain(seed=69420, constant=False)
    ]
).graining(deband)

grain = finalise_clip(grain)


if __name__ == '__main__':
    encoder = X265('x265_settings')
    encoder.resumable = True
    encoder.run_enc(grain, JPBD, qpfile_clip=JPBD.clip_cut)
    # encoder.run_enc(grain[137286:137538], JPBD, qpfile_clip=JPBD.clip_cut[137286:137538])
else:
    stg.output(src)
    stg.output(grain)
    # stg.output(denoise)
    # stg.output(clean)
    # stg.output(hardsub)
