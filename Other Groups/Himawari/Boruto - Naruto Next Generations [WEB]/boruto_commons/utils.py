import inspect
import vapoursynth as vs
from vsutil import depth
from pathlib import Path
from typing import Tuple, Union
from vardautomation import VPath
from lvsfunc.scale import ssim_downsample
from lvsfunc.kernels import BlackManMinLobe


def get_filenames(filename: Union[str, Path]) -> Tuple[str, str]:
    return rf"{str(filename).replace('.mkv', '_sc.log')}", str(get_final_filename(filename))


def get_final_filename(filename: Union[str, Path]) -> VPath:
    return VPath(str(filename).replace('.mkv', '_premux.264'))


# 190~205
def resize_spline(clip: vs.VideoNode) -> vs.VideoNode:
    return depth(clip.resize.Spline64(1280, 720), 16)


# 206~
def resize_ssim(clip: vs.VideoNode) -> vs.VideoNode:
    return depth(ssim_downsample(clip, 1280, 720, kernel=BlackManMinLobe()), 16)


def get_ep_number(filename: str):
    return filename.split('_')[-1].split('.')[0]


def get_default_path(filename: str = None):
    if filename is None:
        filename = inspect.stack()[1].filename
    ep_num = get_ep_number(filename)
    return (
        fr"G:\Drive condivisi\Fansub\_Other\[himawari] Boruto - Naruto Next Generations\{ep_num}"
        fr"\[Crunchyroll] Boruto - Naruto Next Generations - {ep_num} (1080p).mkv"
    )
