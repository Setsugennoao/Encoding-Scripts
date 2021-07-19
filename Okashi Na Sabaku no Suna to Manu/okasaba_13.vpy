import stgfunc as stg
import EoEfunc as eoe
import kagefunc as kgf
import vapoursynth as vs
from okasaba_commons.constants import grain_amount, mitchell

core = vs.core

locipo = stg.src(r"G:\Drive condivisi\Fansub\_Ongoing\Okashi na Sabaku no Suna to Manu\Okashi na Sabaku no Suna to Manu - 13\キャラ@声部 #57 - Locipo（ロキポ） 無料テレビ動画・見逃しネット配信サービス.ts", 16)
clip = locipo[3598:5396]

denoise = eoe.denoise.BM3D(clip, [2.33, 1.35, 1.3], 2, ["vn", "high"])

upscale = stg.upscale.upscale(denoise)

grain = kgf.adaptive_grain(upscale, grain_amount, True, 8.6)

stg.output(clip)
stg.output(mitchell.scale(clip, 1920, 1080))
stg.output(upscale)
stg.output(grain)
