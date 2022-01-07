from math import ceil, floor
import numpy as np
import EoEfunc as eoe
import functools as fct
import vardefunc as vdf
import vapoursynth as vs
from typing import Tuple
from vsutil import get_y, frame2clip
from .filtering import rescale_aa, continuityFixer

core = vs.core


class Crop():
  animation: Tuple[int, int]
  stationary: Tuple[int, int]
  stationary_heights: Tuple[int, int]

  def __init__(self, animation: Tuple[int, int], stationary: Tuple[int, int], stationary_heights: Tuple[int, int]):
    self.animation = animation
    self.stationary = stationary
    self.stationary_heights = stationary_heights


def fixFrameProps(toFix: vs.VideoFrame, fromFix: vs.VideoFrame):
  for key in toFix.props.keys():
    del toFix.props[key]
  for key, value in fromFix.props.items():
    toFix.props[key] = value


def cropperWrapper(clip_in: vs.VideoNode, CROP: Crop):
  def cropper(n, clip):
    if n in range(*CROP.animation):
      return crop_21_9(clip)
    elif CROP.stationary and (n in range(*CROP.stationary)):
      return crop_21_9(clip, CROP.stationary_heights)
    else:
      return clip

  return core.std.FrameEval(clip_in, fct.partial(cropper, clip=clip_in))


def toFrame(clip: vs.VideoNode) -> vs.VideoFrame:
  return clip.get_frame(0)


def crop_21_9(clip: vs.VideoNode, heights: Tuple[int, int] = None):
  reflected = reflect_borders(get_y(clip), heights)
  rescaled_aa = rescale_aa(reflected)
  shuffled = vdf.misc.merge_chroma(rescaled_aa, clip)
  return remove_borders(shuffled)


def copyTo(_from: np.ndarray, _to: vs.VideoFrame, plane: int = 0):
  plane_array = np.asarray(_to.get_write_array(plane))
  np.copyto(plane_array, _from)
  del plane_array


def getArgMax(array: np.ndarray):
  return round(np.average(np.argmax(array, 0)))


def reflect_borders(clip: vs.VideoNode, heights: Tuple[int, int] = None) -> vs.VideoNode:
  if clip.format.color_family != vs.GRAY:
    raise ValueError("GRAY clips only")

  got_heights = heights is not None
  width, height = clip.width, clip.height

  def blankClipFrame(height: int):
    return toFrame(core.std.BlankClip(clip, width, height, length=1))

  def process(n: int, f: vs.VideoFrame):
    array: np.ndarray = eoe.vsnp.frame_to_array(f)[:, :, 0]

    letterbox = array > (25 << 8)

    top, bottom = heights if got_heights else (getArgMax(letterbox), getArgMax(np.flip(letterbox, 1)))

    if top == bottom == 0:
      return f

    cropped = array[top: height - bottom, :]

    f_edgefix = blankClipFrame(cropped.shape[0]).copy()

    copyTo(cropped, f_edgefix)
    del cropped

    f_edgefix = toFrame(continuityFixer(frame2clip(f_edgefix)))

    edgefix = eoe.vsnp.frame_to_array(f_edgefix)[:, :, 0]
    del f_edgefix

    padded = np.pad(edgefix, ((top, bottom), (0, 0)), mode="edge")
    del edgefix

    f_out = f.copy()

    copyTo(padded, f_out)

    fixFrameProps(f_out, f)

    f_out.props["TOP"] = top
    f_out.props["BOTTOM"] = bottom

    return f_out

  return core.std.ModifyFrame(clip, clip, process)


def remove_borders(clip: vs.VideoNode, crop_heights: Tuple[int, int] = None, fill_colors=[4096, 32768, 32768]) -> vs.VideoNode:
  rateo = clip.format.subsampling_h + 1

  def remove_borders_func(n, f: vs.VideoFrame):
    top_y, bottom_y = (f.props["TOP"], f.props["BOTTOM"]) if "TOP" in f.props else (crop_heights or (0, 0))

    if rateo == 1:
      top_uv, bottom_uv = (top_y, bottom_y)
    elif top_y == bottom_y:
      top_uv, bottom_uv = (int(top_y / rateo), int(bottom_y / rateo))
    elif top_y > bottom_y:
      top_uv, bottom_uv = (floor(top_y / rateo), ceil(bottom_y / rateo))
    else:
      top_uv, bottom_uv = (ceil(top_y / rateo), floor(bottom_y / rateo))

    f_out = f.copy()

    def crop_plane(index: int, fill_color: int, top: int, bottom: int):
      array = np.array(f.get_read_array(index), copy=False)
      cropped = array[top: array.shape[0] - bottom, :]
      del array

      padded = np.pad(cropped, ((top, bottom), (0, 0)), mode="constant", constant_values=fill_color)
      del cropped

      plane_array = np.asarray(f_out.get_write_array(index))

      np.copyto(plane_array, padded)
      del plane_array

    crop_plane(0, fill_colors[0], top_y, bottom_y)
    crop_plane(1, fill_colors[1], top_uv, bottom_uv)
    crop_plane(2, fill_colors[2], top_uv, bottom_uv)

    fixFrameProps(f_out, f)

    return f_out

  return core.std.ModifyFrame(clip, clip, remove_borders_func)
