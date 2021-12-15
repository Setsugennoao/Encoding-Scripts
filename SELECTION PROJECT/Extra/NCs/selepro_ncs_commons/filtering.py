import vapoursynth as vs
from vsutil import depth
from debandshit import dumb3kdb
from typing import List, Dict, Any, Union

from .masking import detail_mask
from .utils import get_bits


core = vs.core

# Stolen from Light <3


def masked_f3kdb(
    clip: vs.VideoNode, rad: int = 16, thr: Union[int, List[int]] = 24,
    grain: Union[int, List[int]] = [12, 0], mask_args: Dict[str, Any] = {}
) -> vs.VideoNode:
  deb_mask_args: Dict[str, Any] = dict(brz=(1000, 2750))
  deb_mask_args |= mask_args

  bits, clip = get_bits(clip)

  deband_mask = detail_mask(clip, **deb_mask_args)

  deband = dumb3kdb(clip, radius=rad, threshold=thr, grain=grain, seed=69420)
  deband_masked = core.std.MaskedMerge(deband, clip, deband_mask)
  deband_masked = deband_masked if bits == 16 else depth(deband_masked, bits)
  return deband_masked
