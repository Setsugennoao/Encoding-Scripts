import stgfunc as stg
import kagefunc as kgf
import vapoursynth as vs
from vsutil import plane, depth, join
from boruto_commons.utils import resize_spline, get_default_path, get_filenames
from boruto_commons.scenefilter import filterEnding16, filterPreview, filterTVTokyo
from boruto_commons.constants import FINAL_GRAIN_AMOUNT, ENCODING_x264_ARGS, TV_TOKYO_FRAMES, ED_16_FRAMES
from boruto_commons.filtering import knl_filter, descale_denoiseY_filter, debanding_filter, limit_yuv_filter, bil_downscale

ED_16_FRAMES = ED_16_FRAMES[1]

core = vs.core

filename = get_default_path(__file__)

src = stg.src(filename, 16, matrix_prop=1)

denoised_y, descale_mask = descale_denoiseY_filter(src)
bil_downscale = bil_downscale(src)

clip_y, clip_uv = knl_filter(denoised_y, 'Y'), knl_filter(src, 'UV')

denoised = join([
    resize_spline(clip_y),
    resize_spline(plane(clip_uv, 1)),
    resize_spline(plane(clip_uv, 2))
])

scenefiltered = filterTVTokyo(denoised, bil_downscale, TV_TOKYO_FRAMES)

scenefiltered = filterPreview(scenefiltered, bil_downscale, descale_mask, ED_16_FRAMES)

scenefiltered = filterEnding16(src, scenefiltered, ED_16_FRAMES)

deband = debanding_filter(scenefiltered)

grain = kgf.adaptive_grain(deband, FINAL_GRAIN_AMOUNT, True, 6.5)
grain = depth(grain, 10)
grain = limit_yuv_filter(grain)

if __name__ == '__main__':
  qpfilename, outfilename = get_filenames(filename)

  stg.encode.create_qpfile(src, qpfilename)
  stg.encode.encode(grain, outfilename, False, [], **ENCODING_x264_ARGS(qpfilename))
elif __name__ == '__vapoursynth__':
  grain.set_output()
else:
  stg.output(bil_downscale)
  stg.output(grain)
  stg.output(denoised)
  stg.output(scenefiltered)
  stg.output(deband)
