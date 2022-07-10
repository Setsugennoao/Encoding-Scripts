
from typing import Callable, List

import vapoursynth as vs
from vardefunc import initialise_input, to_444
from vskernels import Bicubic, Bilinear, Catrom, Lanczos, Mitchell, Spline36

core = vs.core


descalers: List[Callable[[vs.VideoNode, int, int], vs.VideoNode]] = [
    Spline36().descale,
    Lanczos().descale,
    Bilinear().descale,
    Bicubic().descale,
    Catrom().descale,
    Mitchell().descale,
]


class Scaling:
    @classmethod
    def upscale_i444(cls, clip: vs.VideoNode) -> vs.VideoNode:
        return to_444(clip, None, None, True)

    @classmethod
    @initialise_input(bits=32)
    def descale(cls, y: vs.VideoNode) -> vs.VideoNode:
        return core.std.Expr(
            [descaler(y, 1600, 900) for descaler in descalers],
            ' '.join([
                'x y z a b c min max min max min',
                'y z a b c max min max min max',
                'z a b c min max min max',
                'a b c max min max',
                'b c min max'
            ])
        )
