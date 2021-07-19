import stgfunc as stg
import havsfunc as haf
import kagefunc as kgf
import vardefunc as vdf
import vapoursynth as vs
from vsutil import depth

# relatively old script...

core = vs.core

src = stg.src(r"E:\Desktop\LL TV\NHK Kouhaku 66th\nhk kouhaku uta gassen 66th nhkg 20151231.ts", 16)
src = src[84000:89014]

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

IVTC = depth(IVTC, 32)

deblock_y = IVTC.deblock.Deblock(34, 10, -2, 0)
deblock_y = deblock_y.deblock.Deblock(15, 3, -2, 0)
deblock_y = deblock_y.deblock.Deblock(10, 7, 7, 0)

deblock_uv = IVTC.deblock.Deblock(30, 10, -2, [1, 2])
deblock_uv = IVTC.deblock.Deblock(20, 6, -1, [1, 2])

light_deblock = IVTC.deblock.Deblock(27, 6, -2)

deblock = vdf.misc.merge_chroma(deblock_y, deblock_uv)

denoise = core.bm3dcuda.BM3D(light_deblock, deblock, 2)

upscale = denoise.resize.Spline64(1920, 1080)
IVTC = IVTC.resize.Spline64(1920, 1080)

grain = kgf.adaptive_grain(upscale, 0.25, True, 1)

grain = depth(grain, 10)

if __name__ == '__vapoursynth__':
  grain.set_output()
else:
  stg.output(src.std.SeparateFields(True).resize.Bicubic(1920, 1080))
  stg.output(IVTC)
  stg.output(grain)
