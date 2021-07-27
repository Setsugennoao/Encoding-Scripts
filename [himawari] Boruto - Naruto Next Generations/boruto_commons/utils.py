import inspect
import vapoursynth as vs
from vsutil import depth
from typing import Tuple


def get_filenames(filename: str) -> Tuple[str, str]:
  return rf"{filename.replace('.mkv', '_sc.log')}", rf"{filename.replace('.mkv', '_premux.mkv')}"


def resize_spline(clip: vs.VideoNode) -> vs.VideoNode:
  return depth(clip.resize.Spline64(1280, 720), 16)


def get_ep_number(filename: str):
  return filename.split('_')[-1].split('.')[0]


def get_default_path(filename: str = None):
  if filename is None:
    filename = inspect.stack()[1].filename
  ep_num = get_ep_number(filename)
  return fr"G:\Drive condivisi\Fansub\_Other\[himawari] Boruto - Naruto Next Generations\{ep_num}\[Crunchyroll] Boruto - Naruto Next Generations - {ep_num} (1080p).mkv"
