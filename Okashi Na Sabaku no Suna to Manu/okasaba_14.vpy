import stgfunc as stg
import kagefunc as kgf
import vapoursynth as vs
from vsutil import depth
from okasaba_commons.constants import grain_amount
from okasaba_commons.filtering import degrain_filter, denoise_filter, knl_filter, dehalo_filter

core = vs.core

locipo = stg.src(r"G:\Drive condivisi\Fansub\_Ongoing\Okashi na Sabaku no Suna to Manu\Okashi na Sabaku no Suna to Manu - 14\キャラ@声部 #58 - Locipo（ロキポ） 無料テレビ動画・見逃しネット配信サービス.ts", 16)
youtube = stg.src(r"G:\Drive condivisi\Fansub\_Ongoing\Okashi na Sabaku no Suna to Manu\Okashi na Sabaku no Suna to Manu - 14\『おかしなさばくのスナとマヌ』第14話「ばんそうこう」-3d4JsV0gAxc.mkv", 16)
locipo, youtube = locipo[3598:5396], youtube[:1798]

yt_degrain = degrain_filter(youtube)
yt_knl = knl_filter(yt_degrain)

lcp_upscale = stg.upscale.upscale(locipo)

fine_detail_mask = stg.mask.generate_detail_mask(lcp_upscale, 0.006)

masked = core.std.MaskedMerge(yt_knl, lcp_upscale, fine_detail_mask)

denoise = denoise_filter(masked, lcp_upscale)

dehalo = dehalo_filter(denoise)

grain = kgf.adaptive_grain(dehalo, grain_amount, True, 8.6)
grain = depth(grain, 10)

if __name__ == '__vapoursynth__':
  grain.set_output()
else:
  stg.output(youtube)
  stg.output(grain)
  stg.output(locipo)
  stg.output(denoise)
  stg.output(lcp_upscale)
  stg.output(fine_detail_mask)
  stg.output(dehalo)
