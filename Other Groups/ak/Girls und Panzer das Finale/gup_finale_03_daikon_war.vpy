import math
from ccd import ccd
import stgfunc as stg
import EoEfunc as eoe
import vapoursynth as vs
from vsutil import depth
from lvsfunc import kernels
from functools import partial
from vardefunc.scale import to_444
from vardefunc.util import finalise_clip
from vsdenoise.bm3d import BM3DCuda, Profile
from vardefunc.noise import Graigasm, AddGrain
from debandshit import placebo_deband, dumb3kdb
from vardautomation import FileInfo, X265, PresetBD

core = vs.core

oycore = stg.oyster.Core()

JPBD = FileInfo(
    r"E:\Desktop\Encoding Sources\BDMVs - ISOs\Girls und Panzer das Finale Ep. 3\GUP_DAS_FINALE_3\BDMV\STREAM\00020.m2ts",
    (24, -24), idx=partial(stg.src, matrix_prop=1), preset=[PresetBD]
)

src = depth(JPBD.clip_cut, 16)

src_444 = to_444(src, 1920, 1080, True, False)

denoisedft = core.dfttest.DFTTest(
    src_444, tbsize=1, slocation=eoe.freq.SLocation([
        0.0, 1.7, 0.35, 1.5, 0.4, 1.8, 0.55, 3, 0.75, 1.9, 1, 1
    ]).interpolated('quadratic').to_list()
)
denoisebm3d = BM3DCuda(src_444, [1.7, 0.45], 1, Profile.NORMAL).clip

degrain_args = dict(tr=1, thSAD=125, thSADC=68)

eoe.dn.CMDegrain(src_444, globalmode=eoe.dn.GlobalMode.WRITE, **degrain_args)

average = core.average.Mean([
    eoe.dn.CMDegrain(
        clip, globalmode=eoe.dn.GlobalMode.READ, **degrain_args
    ) for clip in [denoisedft, denoisebm3d]
])

sloc_high = eoe.freq.SLocation([0.0, 512, 0.15, 4, 0.35, 16, 0.4, 512, 1.0, 512]).interpolated('quadratic')
sloc_low = eoe.freq.SLocation([0.0, 4, 0.15, 512, 0.45, 16, 0.55, 512, 1.0, 512]).interpolated('cubic')

# stg.output(sloc_high.graph())
# stg.output(sloc_low.graph())

merge = eoe.freq.merge_frequency(denoisedft, average, slocation=sloc_high.to_list())
merge = eoe.freq.merge_frequency(denoisebm3d, merge, slocation=sloc_low.to_list())

contra = eoe.misc.ContraSharpening(merge, denoisebm3d)

chroma_denoise = ccd(contra, 4.55)

chroma_denoise = kernels.Catrom(format=vs.YUV420P16).scale(chroma_denoise, 1920, 1080)

deband = core.average.Mean([
    chroma_denoise,
    dumb3kdb(chroma_denoise, 8, 18),
    placebo_deband(chroma_denoise, 8)
])

grain = Graigasm(
    thrs=[x << 8 for x in (42, 80, 128, 176)],
    strengths=[(0.9, 0.02), (0.75, 0.015), (0.55, 0.01), (0.35, 0)],
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
  JPBD.name_clip_output = JPBD.workdir / 'gup_finale_03_daikon_war_v1.265'

  X265('settings/x265_settings_03_daikon_war').run_enc(grain, JPBD)
else:
  stg.output(src)
  stg.output(grain)
  stg.output(chroma_denoise)
  stg.output(deband)
  stg.output(contra)
