import stgfunc as stg
import kagefunc as kgf
import vapoursynth as vs
from vsutil import depth
from okasaba_commons.constants import grain_amount
from okasaba_commons.filtering import degrain_filter, denoise_filter, knl_filter, dehalo_filter

core = vs.core

youtube = stg.src(r"G:\Drive condivisi\Fansub\_Ongoing\Okashi na Sabaku no Suna to Manu\Okashi na Sabaku no Suna to Manu - 15\『おかしなさばくのスナとマヌ』第15話「命名・コアメ」-3tax_lkf81w.mkv", 16)
youtube = youtube[:1798]

yt_degrain = degrain_filter(youtube)
yt_knl = knl_filter(yt_degrain)

denoise = denoise_filter(yt_knl, youtube)

dehalo = dehalo_filter(denoise)

grain = kgf.adaptive_grain(dehalo, grain_amount, True, 8.6)
grain = depth(grain, 10)

if __name__ == '__vapoursynth__':
  grain.set_output()
else:
  stg.output(youtube)
  stg.output(grain)
  stg.output(denoise)
  stg.output(dehalo)
