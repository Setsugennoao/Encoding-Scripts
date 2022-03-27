import lvsfunc as lvf
import vardefunc as vdf
import vapoursynth as vs
from typing import List, Callable
from vardefunc.util import initialise_input

core = vs.core


descalers: List[Callable[[vs.VideoNode, int, int], vs.VideoNode]] = [
    lvf.kernels.Spline36().descale,
    lvf.kernels.Lanczos().descale,
    lvf.kernels.Bilinear().descale,
    lvf.kernels.Bicubic().descale,
    lvf.kernels.Catrom().descale,
    lvf.kernels.Mitchell().descale,
]


class Scaling:
    @classmethod
    def upscale_i444(cls, clip: vs.VideoNode) -> vs.VideoNode:
        return vdf.scale.to_444(clip, None, None, True)

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
