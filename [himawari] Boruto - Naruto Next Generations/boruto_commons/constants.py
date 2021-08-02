import lvsfunc as lvf
from vsutil import get_w

kernel = lvf.kernels.Mitchell()

desc_w, desc_h = (get_w(810), 810)

FINAL_GRAIN_AMOUNT = 0.356


def ENCODING_x264_ARGS(qpfilename: str):
  return dict(
      crf=10, preset="placebo", profile="high444",
      aq_mode=3, aq_strength=0.5, qcomp=0.75,
      bframes=6, ipratio=1.2, pbratio=1.1,
      deblock="-2,-2", ref=8, rc_lookahead=48,
      psy_rd="0.4,0.0", level=4.1,
      colormatrix="bt709", transfer="bt709",
      colorprim="bt709", qpfile=qpfilename,
      output_csp="i444", output_depth=10,
      videoformat="ntsc"
  )


TV_TOKYO_FRAMES = (0, 95)
ED_15_FRAMES = (31517, 33661)

ED_16_FRAMES = [
    (31505, 33661),
    (31504, 33661),
    (31505, 33662),
    (31506, 33663),
    (31506, 33661)
]

ED_17_FRAMES = [
    (31505, 33661),  # 206
    (31506, 33662),  # 207
    (31504, 33662),  # 208
    (31505, 33662),  # 209
]
