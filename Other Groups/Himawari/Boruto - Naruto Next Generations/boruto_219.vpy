import stgfunc as stg
import kagefunc as kgf
import vapoursynth as vs
from vardautomation import FileInfo
from vsutil import plane, depth, join
from boruto_commons.encoding import Encoding
from boruto_commons.utils import resize_ssim, get_default_path
from boruto_commons.constants import FINAL_GRAIN_AMOUNT, TV_TOKYO_FRAMES
from boruto_commons.scenefilter import filterPreview, filterEnding18, filterTVTokyo
from boruto_commons.filtering import knl_filter, descale_denoiseY_filter, debanding_filter, limit_yuv_filter, bil_downscale

ED_18_FRAMES = (31504, 33661)

core = vs.core

CRUNCHY = FileInfo(get_default_path(__file__))

src = depth(CRUNCHY.clip_cut, 16)

denoised_y, descale_mask = descale_denoiseY_filter(src)
bil_downscale = bil_downscale(src)

clip_y, clip_uv = knl_filter(denoised_y, 'Y'), knl_filter(src, 'UV')

denoised = join([
    resize_ssim(clip_y),
    resize_ssim(plane(clip_uv, 1)),
    resize_ssim(plane(clip_uv, 2))
])

scenefiltered = filterTVTokyo(denoised, bil_downscale, TV_TOKYO_FRAMES)

scenefiltered = filterPreview(scenefiltered, bil_downscale, descale_mask, ED_18_FRAMES)

scenefiltered = filterEnding18(src, scenefiltered, ED_18_FRAMES)

deband = debanding_filter(scenefiltered)

grain = kgf.adaptive_grain(deband, FINAL_GRAIN_AMOUNT, True, 6.5)
grain = depth(grain, 10)
grain = limit_yuv_filter(grain)

if __name__ == '__main__':
  brrr = Encoding(CRUNCHY, grain)
  brrr.run()
elif __name__ == '__vapoursynth__':
  grain.set_output()
else:
  stg.output(bil_downscale)
  stg.output(grain)
  stg.output(denoised)
  stg.output(scenefiltered)
  stg.output(deband)
