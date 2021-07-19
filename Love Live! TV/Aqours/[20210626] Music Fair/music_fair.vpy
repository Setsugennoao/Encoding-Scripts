import stgfunc as stg
import havsfunc as haf
import kagefunc as kgf
import vapoursynth as vs
from vsutil import depth

# relatively old script made in 5m or so, was still better than alternatives lol

core = vs.core

src = stg.src(r"E:\Desktop\MUSIC FAIR_202106261800_フジテレビ.ts", 16)
src = src[2176:12558]

IVTC = haf.QTGMC(
    src, Preset="placebo", SubPel=2, SubPelInterp=2,
    BlockSize=4, NNeurons=4, NNSize=6, InputType=0,
    TFF=True, TR0=2, TR1=2, TR2=3,
    EdiMode="EEDI3+NNEDI3", EZDenoise=3,
    MatchEnhance=0.75, NoiseProcess=2,
    GrainRestore=0.4, NoiseRestore=0.2,
    Sigma=1.8, NoiseDeint="Generate",
    Denoiser="dfttest", DenoiseMC=True,
    StabilizeNoise=True
)

out = stg.upscale.upscale(IVTC, 1920, 1080)

grain = kgf.adaptive_grain(out, 0.25, True, 2.5)

grain = depth(grain, 10)

grain.set_output()
