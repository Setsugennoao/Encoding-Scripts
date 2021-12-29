import math
from ccd import ccd
import stgfunc as stg
import EoEfunc as eoe
from xvs import mwcfix
import vapoursynth as vs
from lvsfunc import kernels
from functools import partial
from vsutil import depth, split
from vardefunc.scale import to_444
from vardefunc.util import finalise_clip
from vsdenoise.bm3d import BM3DCuda, Profile
from vardefunc.noise import Graigasm, AddGrain
from debandshit import placebo_deband, dumb3kdb
from vardautomation import FileInfo, X265, FFV1, PresetBD, SelfRunner, RunnerConfig

core = vs.core

oycore = stg.oyster.Core()

JPBD = FileInfo(
    r"E:\Desktop\Encoding Sources\BDMVs - ISOs\Girls und Panzer das Finale Ep. 3\GUP_DAS_FINALE_3\BDMV\STREAM\00002.m2ts",
    (456 + 24, -24), idx=partial(stg.src, matrix_prop=1), preset=[PresetBD]
)

src = depth(JPBD.clip_cut, 16)

src_444 = to_444(src, 1920, 1080, True, False)

fuck_compression = core.average.Mean([
    src_444.deblock.Deblock(quant) for quant in range(16, 32, 4)
])
denoisedft = core.dfttest.DFTTest(
    src_444, tbsize=1, slocation=eoe.freq.SLocation([
        0.0, 1.1, 0.25, 1.5, 0.45, 1.7, 0.7, 1.9, 1, 0.87
    ]).interpolated('quadratic').to_list()
)
denoisebm3d = BM3DCuda(src_444, [1.4, 0.45], 1, Profile.NORMAL).clip
den_rgb = kernels.Catrom(format=vs.RGBS, matrix_in=1).scale(denoisebm3d, 1920, 1080)

degrain_args = dict(tr=1, thSAD=115, thSADC=60)

eoe.dn.CMDegrain(fuck_compression, globalmode=eoe.dn.GlobalMode.WRITE, **degrain_args)

segs_clips = [
    eoe.dn.CMDegrain(
        eoe.misc.ContraSharpening(clip, src_444, 2, 17),
        globalmode=eoe.dn.GlobalMode.READ, **degrain_args
    ) for clip in [denoisedft, denoisebm3d]
]

average = core.average.Mean([denoisebm3d, fuck_compression.rgvs.RemoveGrain(16), core.average.Mean(segs_clips)] + segs_clips)

sloc_high = eoe.freq.SLocation([0.0, 512, 0.15, 4, 0.35, 16, 0.4, 512, 1.0, 512]).interpolated('quadratic')
sloc_low = eoe.freq.SLocation([0.0, 4, 0.15, 512, 0.45, 16, 0.55, 512, 1.0, 512]).interpolated('cubic')

# stg.output(sloc_high.graph())
# stg.output(sloc_low.graph())

merge = eoe.freq.merge_frequency(denoisedft, average, slocation=sloc_high.to_list())
merge = eoe.freq.merge_frequency(denoisebm3d, merge, slocation=sloc_low.to_list())

contra = eoe.misc.ContraSharpening(eoe.misc.ContraSharpening(merge, src_444), denoisebm3d)

contra_rgb = kernels.Catrom(format=vs.RGBS, matrix_in=1).scale(contra, 1920, 1080)

cdenoise = ccd(contra_rgb, 3)
cdenoise = core.std.Expr([cdenoise, contra_rgb, den_rgb], ['x y max z min', 'x', 'x'])

cdenoise = kernels.Catrom(format=vs.YUV444P16, matrix=1).scale(cdenoise, 1920, 1080)

chroma_reconstruct = core.std.Expr([mwcfix(cdenoise), cdenoise, contra], ['x', 'x', 'x y max z max'])

chromafix = chroma_reconstruct.warp.AWarpSharp2(
    thresh=88, blur=3, type=1, depth=4, planes=1
).warp.AWarpSharp2(
    thresh=76, blur=3, type=1, depth=5, planes=2
)


chromafix = core.std.Expr([chromafix, chroma_reconstruct, cdenoise], ['x', 'x', 'x y min z min'])

chromafix = kernels.Catrom(format=vs.YUV420P16).scale(chromafix, 1920, 1080)

sigmaS, sigmaR = 1.45, 5 / 255

den_b = BM3DCuda(chromafix, 7.65, 1).clip

ref = den_b.bilateralgpu.Bilateral(sigmaS, sigmaR)

bidh = den_b.bilateral.Bilateral(ref=ref, sigmaS=sigmaS / 3, sigmaR=sigmaR)

mean_avg = core.average.Mean([chromafix, den_b, bidh])

freqmerge = oycore.FreqMerge(mean_avg, chromafix)

cfixcontra = eoe.misc.ContraSharpening(chromafix, src)

fmergecontra = eoe.misc.ContraSharpening(freqmerge, cfixcontra)

merge = core.std.Expr([
    fmergecontra, bidh, chromafix, freqmerge
], "x y z a min max min y z a max min max z a max max a max")

chroma_denoise = ccd(merge, 4.55)

thr, exp_molt, molt = 20, 3, 0.75
evref = core.std.Expr(
    split(
        merge.resize.Bicubic(
            format=vs.RGBS
        ).std.Prewitt().std.Expr(
            f'x {thr} >= {(1 << 32) - 1} x ?'
        ).std.Median().std.Convolution(
            matrix=[1, 2, 1, 2, 4, 2, 1, 2, 1]
        ).std.RemoveFrameProps('_Matrix')
    ), f'x y + z + 2 16 {exp_molt} + pow * {molt} *', vs.GRAY16
).std.Limiter()

pdeband = placebo_deband(chroma_denoise, 14, 3, 2, 0).std.MaskedMerge(chroma_denoise, evref)

deband = core.average.Mean([
    chroma_denoise,
    dumb3kdb(pdeband, 8, 18),
    dumb3kdb(chroma_denoise, 12, 24),
    dumb3kdb(chroma_denoise, 18, 30),
    dumb3kdb(chroma_denoise, 24, 34),
    placebo_deband(chroma_denoise, 8),
    placebo_deband(chroma_denoise, 12),
    placebo_deband(chroma_denoise, 16)
]).std.MaskedMerge(chroma_denoise, evref)

grain = Graigasm(
    thrs=[x << 8 for x in (42, 80, 128, 176)],
    strengths=[(0.8, 0.02), (0.75, 0.015), (0.5, 0.01), (0.25, 0)],
    sizes=(0.7, 0.65, 0.6, 0.6),
    sharps=(70, 65, 55, 50),
    grainers=[
        AddGrain(seed=420 * 69 * math.pi, constant=False),
        AddGrain(seed=69 * 420 * math.cos(69), constant=False),
        AddGrain(seed=420 * 69 * math.sin(69), constant=True),
        AddGrain(seed=21 * 365 * math.sqrt(69), constant=True)
    ]
).graining(deband)

grain = finalise_clip(grain)

if __name__ == "__main__":
  JPBD.name_clip_output = JPBD.workdir / 'gup_finale_03_v2.265'
#   JPBD.name_clip_output_lossless = JPBD.workdir / 'gup_finale_03_lossless_v1.mkv'
#   JPBD.do_lossless = True

  encoder = X265('settings/x265_settings_03')
  config = RunnerConfig(encoder, FFV1())

  runner = SelfRunner(grain, JPBD, config)

  runner.run()
else:
  stg.output(src)
  stg.output(grain)
  stg.output(chroma_denoise)
  stg.output(deband)
  stg.output(contra)
  stg.output(cdenoise)
  stg.output(chroma_reconstruct)
  stg.output(chromafix)
  stg.output(evref)
