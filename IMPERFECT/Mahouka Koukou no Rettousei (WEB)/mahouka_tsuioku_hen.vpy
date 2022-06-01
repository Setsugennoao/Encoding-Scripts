# this is just a draft bruh, never completed before the BD went out

from ccd import ccd
import stgfunc as stg
import EoEfunc as eoe
import lvsfunc as lvf
import havsfunc as haf
import vapoursynth as vs
from stgfunc import depth
from dataclasses import dataclass
from vsmask.edge import PrewittStd
from lvsfunc.kernels import Catrom
from vardefunc.scale import to_444
from fine_dehalo import fine_dehalo
from stgfunc.utils import replace_squaremask
from vsdenoise.bm3d import BM3DCuda, Profile
from vsutil import Dither, get_y, iterate, split, join, plane


core = vs.core
oycore = stg.oyster.Core()
ssim_kwargs = dict(smooth=((3 ** 2 - 1) / 12) ** 0.5, sigmoid=True, filter_param_a=-1 / 2, filter_param_b=1 / 4)


@dataclass
class ranges:
    ANIMEPLUS = [
        (2210, 2398), (14627, 14770), (33931, 34124), (37288, 37482),
        (51578, 51767), (67961, 68143), (73037, 73225), (89893, 90083)
    ]
    FIX_PATTERN = [
        (77550, 77554), (77566, 77570), (77575, 77580), (77599, 77604),
        (81995, 82000), (82031, 82036), (82103, 82108), (82115, 82120),
        (82139, 82144)
    ]
    REPLACE_TV = [
        (14579, 14626), (67850, 67960)
    ]


def deinterlace(tvcap: vs.VideoNode, web_avg: vs.VideoNode) -> vs.VideoNode:
    matched = core.tivtc.TFM(tvcap, order=1, mode=1, chroma=True)

    qtgmc = haf.QTGMC(
        matched, 'slow', TFF=True, FPSDivisor=2, ChromaEdi='nnedi3',
        EdiMode='EEDI3+NNEDI3', EdiQual=2, NNSize=4, NNeurons=4
    ).tivtc.TDecimate()
    qtgmc = depth(qtgmc, 16)

    qtgmc = qtgmc[:2110] + qtgmc[4507:]
    qtgmc = qtgmc.std.DuplicateFrames(2109).std.DeleteFrames(2114)
    qtgmc = qtgmc[:14578] + qtgmc[1676735:]
    qtgmc = qtgmc.std.DuplicateFrames(14577).std.DeleteFrames(14581)
    qtgmc = qtgmc[:33928] + qtgmc[36805:]
    qtgmc = qtgmc.std.DuplicateFrames(33929)
    qtgmc = qtgmc[:37213] + qtgmc[39372:]
    qtgmc = qtgmc[:37210] + web_avg[37210:37214] + qtgmc[37214:]
    qtgmc = qtgmc[:51502] + qtgmc[53660:]
    qtgmc = qtgmc.std.DuplicateFrames(51501)
    qtgmc = qtgmc[:67850] + qtgmc[70727:]
    qtgmc = qtgmc[:73034] + qtgmc[75192:]

    for dup, rem in ranges.FIX_PATTERN:
        qtgmc = qtgmc.std.DuplicateFrames(dup).std.DeleteFrames(rem)

    qtgmc = qtgmc[:89888] + qtgmc[92046:104297]
    qtgmc = qtgmc.std.DuplicateFrames([89887, 102138] * 2)

    def remove_animeplus(start: int, end: int) -> vs.VideoNode:
        return replace_squaremask(qtgmc, web_avg, (460, 260, 120, 0), (start, end), 3.5)

    for (start, end) in ranges.ANIMEPLUS:
        remove_animeplus(start, end)

    qtgmc = lvf.rfs(qtgmc, web_avg, ranges.REPLACE_TV)

    bs11mask = iterate(qtgmc[67848].resize.Bicubic(format=vs.GRAY16).std.BinarizeMask(
        24 << 8), core.std.Maximum, 7).std.Inflate().bilateral.Gaussian(5)
    bs11mask = core.std.Expr([bs11mask, iterate(bs11mask.bilateral.Gaussian(15),
                             core.std.Maximum, 15)], 'x y +').std.Limiter()

    return qtgmc.std.MaskedMerge(web_avg, bs11mask)


def fake_ass_rescale(clip: vs.VideoNode) -> vs.VideoNode:
    y = get_y(clip)
    finedehalo = fine_dehalo(y, rx=2.1, ry=2.1, darkstr=0, brightstr=0.86)

    sigmaS, sigmaR = 1.5, 5 / 255

    den = BM3DCuda(finedehalo, 8, radius=1).clip

    ref = den.bilateral.Bilateral(sigmaS=sigmaS, sigmaR=sigmaR)
    bidh = den.bilateral.Bilateral(ref=ref, sigmaS=sigmaS / 3, sigmaR=sigmaR)

    dehalo = core.std.Expr([finedehalo, bidh, y], "x y min z min")

    lineart_mask = PrewittStd().edgemask(y.rgsf.RemoveGrain(3).rgsf.RemoveGrain(3), 0.1)
    lineart_mask = lineart_mask.std.Maximum().std.Minimum().std.Minimum().std.BoxBlur().std.Deflate().std.Expr('x 10 *')

    masked_dehalo = dehalo.std.MaskedMerge(finedehalo, lineart_mask)

    y_32 = depth(masked_dehalo, 32, dither_type=Dither.NONE)

    descale = Catrom().descale(y_32, 1488, 837)

    return depth(descale, 16)


crunchy = stg.src(r".\Source\Mahouka Koukou no Rettousei - Tsuioku-hen (CR).mkv", 16, matrix_prop=1)
funi = stg.src(r".\Source\Mahouka Koukou no Rettousei - Tsuioku-hen (FUNi).mkv", 16, matrix_prop=1)[289:]
tvcap = stg.src(r".\Source\Mahouka Koukou no Rettousei - Tsuioku-hen (BS11).dgi", 8, matrix_prop=1)[6725:]
deint_masked = deinterlace(tvcap, core.average.Mean([funi, crunchy]))

loc_fixes = map(lambda x: x.resize.Bicubic(chromaloc_in=1, chromaloc=0), [crunchy, funi])

srcs_444 = list(map(lambda x: to_444(x, 1920, 1080, True, False), loc_fixes))

deblocks = [[src.deblock.Deblock(quant) for quant in range(16, 32, 4)] for src in srcs_444]

averages = map(core.average.Mean, deblocks)

averages = [
    eoe.misc.ContraSharpening(den, src)
    for den, src in zip(averages, srcs_444)
]

cdenoise = [ccd(den, thr) for den, thr in zip(averages, range(4, 7))]

bm3d_sigmas = [[1.3, 0.8], [1.7, 1], [2, 1.6]]

denoisebm3d = [BM3DCuda(fcomp, sigma, 1, Profile.NORMAL).clip for fcomp, sigma in zip(cdenoise, bm3d_sigmas)]


merge_web = oycore.FreqMerge(denoisebm3d[0], denoisebm3d[1])
avg_web = core.average.Mean([merge_web, *denoisebm3d[:2]])

denoisebm3d[2] = join(
    [(planes := split(denoisebm3d[2]))[0]] + [eoe.misc.ContraSharpening(dplane, plane(avg_web, i)) for dplane, i in zip(planes[1:], [1, 2])]
)

merge_tv = oycore.FreqMerge(denoisebm3d[2], avg_web)

merges = [merge_web, merge_tv]

denoise_dft = merges

denoise_dft = [
    core.dfttest.DFTTest(
        src, tbsize=1, slocation=eoe.freq.SLocation([
            0.0, 0.8, 0.25, 1, 0.45, 1.3, 0.76, 1.42, 1, 0.65
        ]).interpolated('quadratic').to_list()
    ) for src in merges
]

degrain_args = dict(tr=1, thSAD=165, thSADC=85)

eoe.dn.CMDegrain(denoisebm3d[0], globalmode=eoe.dn.GlobalMode.WRITE, **degrain_args)

segs_clips = [
    eoe.dn.CMDegrain(avg_web, globalmode=eoe.dn.GlobalMode.READ, **degrain_args)
    for clip in denoise_dft
]

average = core.average.Mean([*merges, *segs_clips] + segs_clips)

sloc_high = eoe.freq.SLocation([0.0, 512, 0.15, 4, 0.35, 16, 0.4, 512, 1.0, 512]).interpolated('quadratic')
sloc_low = eoe.freq.SLocation([0.0, 4, 0.15, 512, 0.45, 16, 0.55, 512, 1.0, 512]).interpolated('cubic')

merge = eoe.freq.merge_frequency(merge_web, average, slocation=sloc_high.to_list())
merge = eoe.freq.merge_frequency(merge_tv, merge, slocation=sloc_low.to_list())

stg.output(srcs_444[0])
stg.output(srcs_444[1])
stg.output(srcs_444[2])
# stg.output(average)
# core.imwri.Write(merge[92739].resize.Bicubic(format=vs.RGBS), 'PNG', 'image%d.png', 0).get_frame(0)
