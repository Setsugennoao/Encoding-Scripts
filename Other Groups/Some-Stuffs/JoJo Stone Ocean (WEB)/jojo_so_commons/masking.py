import vapoursynth as vs
from vsmask.edge import ExLaplacian4, PrewittStd, region_mask
from vsutil import fallback

core = vs.core


class ExLaplaWitt(ExLaplacian4):
    def _compute_mask(self, clip: vs.VideoNode) -> vs.VideoNode:
        exlaplacian4 = super()._compute_mask(clip)
        prewitt = PrewittStd().edgemask(clip)
        mask = core.std.Expr((exlaplacian4, prewitt), 'x y max')
        return region_mask(mask, right=2).fb.FillBorders(right=2)


def mt_xxpand_multi(
    clip, sw=1, sh=None, mode='square', planes=None,
    start=0, M__imum=core.std.Maximum, **params
):
    sh = fallback(sh, sw)
    planes = (
        list(range(clip.format.num_planes)) if planes is None
        else [planes] if isinstance(planes, int) else planes
    )

    if mode == 'ellipse':
        coordinates = [[1] * 8, [0, 1, 0, 1, 1, 0, 1, 0], [0, 1, 0, 1, 1, 0, 1, 0]]
    elif mode == 'losange':
        coordinates = [[0, 1, 0, 1, 1, 0, 1, 0]] * 3
    else:
        coordinates = [[1] * 8] * 3

    clips = [clip]

    end = min(sw, sh) + start

    for x in range(start, end):
        clips += [
            M__imum(clips[-1], coordinates=coordinates[x % 3], planes=planes, **params)
        ]

    for x in range(end, end + sw - sh):
        clips += [
            M__imum(clips[-1], coordinates=[0, 0, 0, 1, 1, 0, 0, 0], planes=planes, **params)
        ]

    for x in range(end, end + sh - sw):
        clips += [
            M__imum(clips[-1], coordinates=[0, 1, 0, 0, 0, 0, 1, 0], planes=planes, **params)
        ]

    return clips
