import inspect
import muvsfunc as mvsf
import vapoursynth as vs
from vsutil import depth
from typing import Tuple
from pathlib import Path


def get_filenames(filename: str) -> Tuple[str, str]:
  return rf"{filename.replace('.mkv', '_sc.log')}", str(get_final_filename(filename))


def get_final_filename(filename: str) -> Tuple[str, str]:
  return Path(filename.replace('.mkv', '_premux.mkv'))


# 190~205
def resize_spline(clip: vs.VideoNode) -> vs.VideoNode:
  return depth(clip.resize.Spline64(1280, 720), 16)


# 206~
def resize_ssim(clip: vs.VideoNode) -> vs.VideoNode:
  return depth(mvsf.SSIM_downsample(clip, 1280, 720, 0, kernel='Spline64'), 16)


def get_ep_number(filename: str):
  return filename.split('_')[-1].split('.')[0]


def get_default_path(filename: str = None):
  if filename is None:
    filename = inspect.stack()[1].filename
  ep_num = get_ep_number(filename)
  return r"G:\Drive condivisi\Fansub\_Other\[himawari] Boruto - Naruto Next Generations\%s\[Crunchyroll] Boruto - Naruto Next Generations - %s (1080p).mkv" % (ep_num, ep_num)
