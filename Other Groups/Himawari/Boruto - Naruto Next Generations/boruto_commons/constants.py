import lvsfunc as lvf
from vsutil import get_w

kernel = lvf.kernels.Mitchell()

desc_w, desc_h = (get_w(810), 810)

FINAL_GRAIN_AMOUNT = 0.356

TV_TOKYO_FRAMES = (0, 95)
