import math
import lvsfunc as lvf

descale_w, descale_h = 1500, 844
kernel = lvf.kernels.Bicubic(0.33, 0.33)
final_grain_amount = math.pi / 20
