import stgfunc as stg
import lvsfunc as lvf
import kagefunc as kgf
import havsfunc as haf
import vapoursynth as vs
from vsutil import depth
from kanojo_commons.constants import final_grain_amount
from kanojo_commons.crop_21_9 import Crop, cropperWrapper
from kanojo_commons.masking import getBandingMask, getDebandMask, lineMasking
from kanojo_commons.filtering import rescale_aa, debanding, denoising, temp_degrain

# vspipe --y4m --progress kanojo_mo_kanojo_12.vpy - | ffmpeg -i - -crf 15
# -preset veryslow -c:v hevc -x265-params
# qpfile="E:\Desktop\kanojo_mo_kanojo_12_sc.log" "E:\Desktop\out_12.mkv"

# inclusive
CROP_IN = Crop((19837, 19871), (19872, 22930), (108, 108))
CROP_OUT = Crop((22931, 22964), None, None)

core = vs.core

clip = stg.src(r"G:\Il mio Drive\Anime\Settimanali\[SubsPlease] Kanojo mo Kanojo - 12 (1080p) [6E59D6D3].mkv", 16)

rescaled_aa_a = rescale_aa(clip)
rescaled_aa_b = rescale_aa(clip, maskAA=True)

rescaled_aa = lvf.rfs(rescaled_aa_a, rescaled_aa_b, [
    (115, 216), (242, 330), (446, 540), (554, 654),
    (668, 768), (783, 882), (896, 995), (1023, 1122),
    (1149, 1243), (1257, 1357), (1371, 1471),
    (1485, 1585), (1616, 1700), (1747, 1825),
    (32600, 32666), (32681, 32792), (32813, 32883),
    (32897, 32967), (33017, 33087), (33185, 33255),
    (33269, 33339), (33348, 33423), (33437, 33501),
    (33545, 33615), (33752, 33846), (33866, 33948),
    (33962, 34044), (34094, 34164), (34190, 34260),
    (34284, 34362), (34404, 34482)
])

aa_masked = rescaled_aa

aa_masked = cropperWrapper(
  cropperWrapper(rescaled_aa, CROP_OUT), CROP_IN
)

ref = temp_degrain(aa_masked)

denoise, denoise_y = denoising(aa_masked, ref)

denoise = lvf.rfs(denoise, stg.denoise.KNLMeansCL(ref, sigma=[1.24, 1.15]), [
    (1034, 1261), (3911, 4003), (4511, 4612), (5319, 5481),
    (5979, 6046), (6575, 6766), (7050, 7258), (18561, 18638),
    (18771, 18932), (31352, 31414)
])

denoise = lvf.rfs(denoise, clip, [
    (25961, 26239), (26367, 26494)
])

deband, deband_y = debanding(denoise, ref)

denoise_y_dark = haf.FastLineDarkenMOD(denoise_y, strength=35)

lines = lineMasking(denoise_y, denoise_y_dark)

debanding_mask = getDebandMask(lines)

deband_dark = core.std.MaskedMerge(deband, denoise, debanding_mask)

merge = core.std.MaskedMerge(deband, deband_dark, getBandingMask(deband_y))

grain = kgf.adaptive_grain(merge, final_grain_amount, True, 8)
grain = depth(grain, 10)

if __name__ == '__vapoursynth__':
  grain.set_output()
else:
  stg.output(clip)
  stg.output(grain)
  stg.output(aa_masked)
  stg.output(denoise)
  stg.output(deband)
  stg.output(lines)
  stg.output(debanding_mask)
