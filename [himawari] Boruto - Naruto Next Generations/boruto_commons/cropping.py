import kagefunc as kgf
import vapoursynth as vs
from .classes import Crop
from math import ceil, floor

core = vs.core


def crop_resize_to_720p(clip: vs.VideoNode, crop: Crop) -> vs.VideoNode:
  clip1080 = clip.resize.Spline64(1920, 1080)
  cropped = clip1080.std.Crop(crop.left, crop.right, crop.top, crop.bottom)

  sides = crop.right != crop.left != 0

  args = (2 if sides else 0, 5) * 2

  edgefixed = cropped.edgefixer.ContinuityFixer(*args) if crop.edgeFixing else cropped

  top, bottom = crop.top * 720 / 1080, crop.bottom * 720 / 1080
  if top > bottom:
    top, bottom = ceil(top), floor(bottom)
  else:
    top, bottom = floor(top), ceil(bottom)

  left, right = crop.left * 1280 / 1920, crop.right * 1280 / 1920
  if left > right:
    left, right = ceil(left), floor(right)
  else:
    left, right = floor(left), ceil(right)

  resized = edgefixed.resize.Spline64(1280 - left - right, 720 - top - bottom)
  padded = resized.std.AddBorders(left, right, top, bottom)

  padded = padded.resize.Spline64(1280, 720)

  bottom_clip_h = 129 * 720 / 1080

  return core.std.MaskedMerge(
      padded, clip,
      kgf.squaremask(clip, 1280, bottom_clip_h, 0, 720 - bottom_clip_h)
  )
