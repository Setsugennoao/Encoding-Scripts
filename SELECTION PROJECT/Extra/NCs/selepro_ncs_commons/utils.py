import lvsfunc as lvf
import vapoursynth as vs
from typing import Tuple
from vsutil import depth, get_depth
from stgfunc.utils import replace_squaremask


def get_bits(clip: vs.VideoNode, expected_depth: int = 16) -> Tuple[int, vs.VideoNode]:
  return (bits := get_depth(clip)), depth(clip, expected_depth) if bits != expected_depth else clip


def freeze_replace_mask(
    mask: vs.VideoNode, insert: vs.VideoNode,
    mask_params: Tuple[int, int, int, int], frame: int, frame_range: Tuple[int, int]
) -> vs.VideoNode:
  masked_insert = replace_squaremask(mask[frame], insert[frame], mask_params)
  return lvf.rfs(mask, masked_insert * mask.num_frames, frame_range)
