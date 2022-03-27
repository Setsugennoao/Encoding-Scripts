import math
import vapoursynth as vs


core = vs.core


def MMFilter(clips, mode='min', planes=None):
    planes = (
        list(range(clips[0].format.num_planes)) if planes is None
        else [planes] if isinstance(planes, int) else planes
    )
    expr = 'x y - x z - * 0 < {} x y - abs x z - abs < {} ? ?'.format(
        *(['z', 'z y'] if mode == 'max' else ['x', 'y z'])
    )
    expr = [expr if i in planes else '' for i in range(clips[0].format.num_planes)]

    return core.std.Expr(clips, expr)


def csharp(flt, src, str=20):
    np = flt.format.num_planes
    blur = flt.rgsf.RemoveGrain(str)
    return core.std.Expr([
        flt, src, blur
    ], [
        'x dup + z - x y min max x y max min', '', ''
    ][:np])


def unsharp(clip, radius=1, strength=math.log2(3), custom=None):
    if callable(custom):
        return clip.std.MergeDiff(clip.std.MakeDiff(custom(clip)))

    strength = max(1e-6, min(strength, math.log2(3)))

    weight = 0.5 ** strength / ((1 - 0.5 ** strength) / 2)

    # find matrix with least rounding error when using integer clips
    if clip.format.sample_type == 0:
        all_matrices = [[x] for x in range(1, 1024)]
        for x in range(1023):
            while len(all_matrices[x]) < radius * 2 + 1:
                all_matrices[x].append(all_matrices[x][-1] / weight)
        error = [sum([abs(x - round(x)) for x in matrix[1:]]) for matrix in all_matrices]
        matrix = [round(x) for x in all_matrices[error.index(min(error))]]
    else:
        matrix = [1]
        while len(matrix) < radius * 2 + 1:
            matrix.append(matrix[-1] / weight)

    matrix = [
        matrix[x] for x in [
            (2, 1, 2, 1, 0, 1, 2, 1, 2),
            (4, 3, 2, 3, 4, 3, 2, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1, 2, 3, 4, 3, 2, 3, 4)
        ][radius - 1]
    ]

    return clip.std.MergeDiff(clip.std.MakeDiff(clip.std.Convolution(matrix)))


def pre_aa(clip, radius=1, strength=math.log2(3), pp=None, **nnedi3_params):

    sharp = unsharp(clip, radius=radius, strength=strength)

    nedi = clip.znedi3.nnedi3(3, **nnedi3_params).std.SeparateFields(True)
    nedi = nedi.std.SelectEvery(4, [0, 3]).std.DoubleWeave()[::2]

    if callable(pp):
        nedi = pp(nedi)

    clip = core.std.Expr([sharp, clip, nedi], 'x y z max min y z min max').std.Transpose()

    sharp = unsharp(clip, radius=radius, strength=strength)

    nedi = clip.znedi3.nnedi3(3, **nnedi3_params).std.SeparateFields(True)
    nedi = nedi.std.SelectEvery(4, [0, 3]).std.DoubleWeave()[::2]

    if callable(pp):
        nedi = pp(nedi)

    return core.std.Expr([sharp, clip, nedi], 'x y z max min y z min max').std.Transpose()
