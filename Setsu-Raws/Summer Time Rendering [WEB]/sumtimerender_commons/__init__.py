from typing import List, Any
from lvsfunc.types import Range
from vardautomation import X265

from .utils import *  # noqa
from .filtering import *  # noqa


def get_encoder(
    ED_RANGES: List[Range],
    MEDIUM_GRAIN_BUT_IDK_MAN_THE_MOTION_BLOCKS_ARE_DYING: List[Range],
    SUPER_COARSE_GRAINY_WTF_KILL_THIS_STUDIO_PLEASE_RANGES: List[Range],
    **kwargs: Any
) -> X265:
    zones = {
        ranges: {'b': 0.75} for ranges in MEDIUM_GRAIN_BUT_IDK_MAN_THE_MOTION_BLOCKS_ARE_DYING
    } | {
        ranges: {'b': 0.65} for ranges in SUPER_COARSE_GRAINY_WTF_KILL_THIS_STUDIO_PLEASE_RANGES
    }

    if ED_RANGES:
        zones |= {
            ED_RANGES: {'b': 0.5}
        }

    encoder = X265('sumtimerender_commons/x265_settings', zones, **kwargs)

    encoder.resumable = True

    return encoder
