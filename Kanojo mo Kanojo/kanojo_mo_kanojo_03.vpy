import stgfunc as stg
import kagefunc as kgf
import havsfunc as haf
import vapoursynth as vs
from vsutil import depth
from kanojo_commons.crop_21_9 import cropperWrapper
from kanojo_commons.constants import final_grain_amount
from kanojo_commons.filtering import rescale_aa, debanding, denoising, temp_degrain
from kanojo_commons.masking import binarizeMask, getBandingMask, getDebandMask, lineMasking, getCreditMask

# vspipe --y4m --progress kanojo_mo_kanojo_03.vpy - | ffmpeg -i - -c:v ffv1
# -coder 1 -context 0 -g 1 -level 3 -threads 4 -slices 24 -slicecrc 1
# "E:\Desktop\KanojoMoKanojo_03_lossless.mkv"

# inclusive
CROP = None
MASK_PATH = r".\masks\mask_03"
ED_FRAMES = (32656, 34812)

core = vs.core

clip = stg.src(r"G:\Il mio Drive\Anime\Settimanali\Kanojo mo Kanojo\[CRUNCHY] Kanojo mo Kanojo - 03 [1080p][MULTI SUB].mkv", 16)
NCED = stg.src(r".\kanojo_commons\NCED\NC_ED_WEB.mkv", 16)

rescaled_aa, rescaled = rescale_aa(clip, (clip, ED_FRAMES), True)

aa_masked = rescaled_aa

if CROP is not None:
  aa_masked = cropperWrapper(aa_masked, CROP)

if MASK_PATH:
  aa_masked = stg.mask.manual_masking(aa_masked, clip, MASK_PATH, binarizeMask)

if ED_FRAMES and NCED:
  aa_masked = core.std.MaskedMerge(aa_masked, rescaled, getCreditMask(clip, NCED, ED_FRAMES[0]))

ref = temp_degrain(aa_masked)

denoise, denoise_y = denoising(aa_masked, ref)
deband, deband_y = debanding(denoise, ref)

denoise_y_dark = haf.FastLineDarkenMOD(denoise_y, strength=50)

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
