import lvsfunc as lvf
import stgfunc as stg
import havsfunc as haf
import kagefunc as kgf
import vapoursynth as vs
from vsutil import depth, get_y
from puchimikku_commons.constants import grain_amount, COMMON_MASKS_PATH
from puchimikku_commons.masking import get_detail_mask, get_linemasks
from puchimikku_commons.filtering import debanding_filter, degrain_filter, denoise_filter, knl_filter, sraa_filter

core = vs.core
core.max_cache_size = 6 * 2 ** 10

HEAVY_KNL_FRAMES = [(0, 5), (3997, 4109), (4143, 4246)]

amazon = stg.src(r"E:\Desktop\D4DJ Petit Mix - 19\Amazon-ぷっちみく♪ D4DJ Petit Mix_S01E19_戦慄令嬢～どうあがいても…絶叫～.mp4", 16)

clip = amazon.edgefixer.ContinuityFixer(3, 3, 3, 3)
clip_y = get_y(clip)

degrain = degrain_filter(clip)

light_knl, heavy_knl = knl_filter(clip, degrain)

denoise, denoise_y = denoise_filter(light_knl, heavy_knl)

denoise_y_dark = haf.FastLineDarkenMOD(denoise_y, 60)

noise_mask = kgf.adaptive_grain(denoise_y_dark, show_mask=True)

degrain_masked = core.std.MaskedMerge(degrain, clip, get_detail_mask(clip_y))

denoise_masked = lvf.rfs(
    core.std.MaskedMerge(degrain_masked, denoise, noise_mask),
    heavy_knl,
    HEAVY_KNL_FRAMES
)

sraa, sraa_y = sraa_filter(denoise_masked)

lineart_hard, lineart_light = get_linemasks(sraa_y)

sraa_masked = core.std.MaskedMerge(denoise_masked, sraa, lineart_hard)
sraa_masked = lvf.rfs(
    sraa_masked,
    core.std.MaskedMerge(denoise_masked, sraa_masked, lineart_light),
    HEAVY_KNL_FRAMES
)

deband = debanding_filter(sraa_masked)
deband = stg.mask.manual_masking(deband, light_knl, COMMON_MASKS_PATH)

grain = kgf.adaptive_grain(deband, grain_amount, True, 12)

grain = depth(grain, 10)

if __name__ == '__vapoursynth__':
  grain.set_output()
else:
  stg.output(clip)
  stg.output(grain)
  stg.output(denoise_masked)
  stg.output(sraa_masked)
  stg.output(light_knl)
  stg.output(heavy_knl)
  stg.output(deband)
