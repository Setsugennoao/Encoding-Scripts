import stgfunc as stg
import kagefunc as kgf
import vapoursynth as vs
import vardefunc as vdf
from vsutil import iterate, depth

core = vs.core


def tcanny(clip: vs.VideoNode, thr: float):
  gaussian = clip.bilateral.Gaussian(1)
  msrcp = core.retinex.MSRCP(gaussian, sigma=[50, 200, 350], upper_thr=thr)
  return msrcp.tcanny.TCannyCL(mode=1, sigma=1).std.Minimum(coordinates=[1, 0, 1, 0, 0, 1, 0, 1])


def binarizeMask(mask: vs.VideoNode):
  binarized = mask.std.Binarize(20 << mask.format.bits_per_sample - 8)
  expanded = iterate(binarized, core.std.Maximum, 2)
  return iterate(expanded, core.std.Inflate, 2)


def lineMasking(y: vs.VideoNode, y_dark: vs.VideoNode) -> vs.VideoNode:
  return core.std.Expr([
      kgf.kirsch(y_dark),
      tcanny(y, 0.000125),
      tcanny(y, 0.0045),
      stg.mask.generate_detail_mask(y_dark, 0.013)
  ], "x y + z a + +")


def getDebandMask(lineMask: vs.VideoNode) -> vs.VideoNode:
  deband_mask = core.std.Binarize(lineMask, 23 << 8)
  deband_mask = iterate(deband_mask, core.std.Maximum, 3)
  return iterate(deband_mask, core.std.Deflate, 2)


def getBandingMask(y: vs.VideoNode) -> vs.VideoNode:
  stats = y.std.PlaneStats()
  return core.adg.Mask(stats, 75)


def getCreditMask(clip: vs.VideoNode, ncclip: vs.VideoNode, start_frame: int) -> vs.VideoNode:
  blur_src = core.bilateral.Gaussian(clip, sigma=2)
  blur_NCED = core.bilateral.Gaussian(ncclip, sigma=2)

  blur_srced = blur_src[start_frame:start_frame + ncclip.num_frames]

  ed_mask = vdf.dcm(blur_src, blur_srced, blur_NCED, start_frame=start_frame, thr=25 << 8, prefilter=True)

  credit_mask = depth(ed_mask, 16)
  credit_mask = iterate(credit_mask, core.std.Maximum, 3)

  return credit_mask.std.Inflate().std.Binarize()
