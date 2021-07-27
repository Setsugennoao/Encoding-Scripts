import stgfunc as stg
import lvsfunc as lvf
import kagefunc as kgf
import vapoursynth as vs
from vsutil import plane, depth, join
from boruto_commons.utils import resize_spline
from boruto_commons.constants import TV_TOKYO_FRAMES, ED_15_FRAMES
from boruto_commons.scenefilter import filterEnding15, filterPreview, filterTVTokyo
from boruto_commons.filtering import knl_filter, descale_denoiseY_filter, debanding_filter, limit_yuv_filter, bil_downscale


core = vs.core

src = stg.src(r"E:\Desktop\Boruto\190\[SubsPlease] Boruto - Naruto Next Generations - 190 (1080p) [74527B94].mkv", 16, matrix_prop=1)

denoised_y, descale_mask = descale_denoiseY_filter(src)
bil_downscale = bil_downscale(src)

clip_y, clip_uv = knl_filter(denoised_y, 'Y'), knl_filter(src, 'UV')

denoised = join([
    resize_spline(clip_y),
    resize_spline(plane(clip_uv, 1)),
    resize_spline(plane(clip_uv, 2))
])

scenefiltered = filterTVTokyo(denoised, bil_downscale, TV_TOKYO_FRAMES)

scenefiltered = filterPreview(scenefiltered, bil_downscale, descale_mask, ED_15_FRAMES)

scenefiltered = lvf.rfs(scenefiltered, filterEnding15(denoised, ED_15_FRAMES), ED_15_FRAMES)

deband = debanding_filter(scenefiltered)

grain = kgf.adaptive_grain(deband, 0.356, True, 6.5)
grain = depth(grain, 10)
grain = limit_yuv_filter(grain)

if __name__ == '__vapoursynth__':
  grain.set_output()
else:
  stg.output(bil_downscale)
  stg.output(grain)
  stg.output(denoised)
  stg.output(scenefiltered)
  stg.output(deband)
