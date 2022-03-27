from ccd import ccd
import stgfunc as stg
import vapoursynth as vs
from vardautomation import FileInfo
from vsutil import plane, depth, join
from vardefunc.util import finalise_clip
from boruto_commons.encoding import Encoding
from boruto_commons.utils import resize_ssim, get_default_path
from boruto_commons.constants import FINAL_GRAIN_AMOUNT, TV_TOKYO_FRAMES
from boruto_commons.scenefilter import filterPreview, filterEnding19, filterTVTokyo
from boruto_commons.filtering import (
    descale_denoiseY_filter, debanding_filter, bil_downscale, adaptive_grain, knl_filter
)

ED_19_FRAMES = (31505, 33661)

core = vs.core

CRUNCHY = FileInfo(get_default_path(__file__))

src = depth(CRUNCHY.clip_cut, 16)

denoised_y, descale_mask = descale_denoiseY_filter(src)
bil_downscale = bil_downscale(src)

clip_y, clip_uv = knl_filter(denoised_y, 'Y'), ccd(src, 5)

denoised = join([
    resize_ssim(clip_y),
    resize_ssim(plane(clip_uv, 1)),
    resize_ssim(plane(clip_uv, 2))
])

scenefiltered = filterTVTokyo(denoised, bil_downscale, TV_TOKYO_FRAMES)

scenefiltered = filterPreview(scenefiltered, bil_downscale, descale_mask, ED_19_FRAMES)

scenefiltered = filterEnding19(src, scenefiltered, ED_19_FRAMES)

deband = debanding_filter(scenefiltered)

grain = adaptive_grain(deband, FINAL_GRAIN_AMOUNT, 6.5)
grain = finalise_clip(grain)

if __name__ == '__main__':
    brrr = Encoding(CRUNCHY, grain).run()
elif __name__ == '__vapoursynth__':
    grain.set_output()
else:
    stg.output(bil_downscale)
    stg.output(grain)
    stg.output(denoised)
    stg.output(scenefiltered)
    stg.output(deband)
