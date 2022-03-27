import stgfunc as stg
import vapoursynth as vs
from vsutil import get_y
from lvsfunc.kernels import Catrom
from stgfunc.utils import replace_squaremask
from vardautomation import X265, FileInfo, PresetWEB, get_vs_core

from takagi_san_commons.filtering import TakagiSanSanFiltering

core = get_vs_core(range(0, vs.core.num_threads, 2))
catrom = Catrom()

fileinfo = FileInfo(
    r".\Source\01\Karakai Jouzu no Takagi-san S3 - 01 (Amazon dAnime VBR 1080p).mkv", preset=[PresetWEB]
)

sources = [
    stg.src(r".\Source\01\Karakai Jouzu no Takagi-san S3 - 01 (Amazon dAnime CBR 1080p).mkv", 16, matrix_prop=1),
    stg.src(r".\Source\01\Karakai Jouzu no Takagi-san S3 - 01 (Amazon dAnime VBR 1080p).mkv", 16, matrix_prop=1),
    stg.src(r".\Source\01\Karakai Jouzu no Takagi-san S3 - 01 (Netflix VBR 1080p).mkv", 16, matrix_prop=1)[24:-24]
]

sources = [source[31838:] for source in sources]
sources = [*sources[:2], core.average.Mean(sources), sources[2]]

nced_sources = [
    stg.src(r".\Source\NCED01\TVアニメ『からかい上手の高木さん』ノンクレジットED「夢で逢えたら」／高木さん(CV -高橋李依)-8cQCn63acJ0.webm", 16, matrix_prop=1),
    stg.src(r".\Source\NCED01\TVアニメ『からかい上手の高木さん』ノンクレジットED「夢で逢えたら」／高木さん(CV -高橋李依)-8cQCn63acJ0.mp4", 16, matrix_prop=1)
]

amazon_cbr = sources[1]
ncop_avg = core.average.Mean(nced_sources)

black_gray = get_y(amazon_cbr).std.BlankClip()
white_gray = black_gray.std.Invert()


def get_ncop(nced: vs.VideoNode, ep_ed: vs.VideoNode) -> vs.VideoNode:
    pink = core.std.BlankClip(ep_ed, color=[48896, 32768, 38912])

    # Ending dehardsubbing
    ep_ed = replace_squaremask(ep_ed, pink, (1165, 1080, 0, 0), [(505, 847), (1312, 1581)])
    ep_ed = replace_squaremask(ep_ed, pink, (1155, 1080, 765, 0), [(873, 1286), (1607, 1969)])

    ed_static = core.average.Mean([ep_ed[f] for f in [
        91, *range(167, 173 + 1), *range(249, 255 + 1), 331, 1988
    ]]) * ep_ed.num_frames

    ep_ed = replace_squaremask(ep_ed, ed_static, (500, 500, 300, 300), (92, 330), 1.4)

    ep_ed = replace_squaremask(ep_ed, ed_static, (383, 320, 105, 56), (338, 479), 1.4)
    ep_ed = replace_squaremask(ep_ed, ed_static, (837, 168, 109, 383), (338, 479), 1.4)
    ep_ed = replace_squaremask(ep_ed, ed_static, (1091, 474, 101, 560), (338, 479), 1.4)
    # should be yt, but it's so shit (wow), that moving details (petals) get nuked so uhhh, fuck it

    ep_ed = replace_squaremask(ep_ed, ed_static, (190, 73, 991, 440), (338, 479), 1.4)

    ep_ed = replace_squaremask(ep_ed, ed_static, (569, 237, 240, 420), (1989, 2062), 1)

    ep_ed = replace_squaremask(ep_ed, nced, (83, 38, 1094, 735), (2070, None), 0.25)
    ep_ed = replace_squaremask(ep_ed, nced, (197, 42, 1231, 733), (2070, None), 0.25)
    ep_ed = replace_squaremask(ep_ed, nced, (243, 73, 1452, 699), (2070, None), 0.25)
    ep_ed = replace_squaremask(ep_ed, nced, (83, 41, 1096, 830), (2070, None), 0.25)
    ep_ed = replace_squaremask(ep_ed, nced, (553, 44, 1227, 828), (2070, None), 0.25)
    ep_ed = replace_squaremask(ep_ed, nced, (770, 38, 1042, 981), (2070, None), 0.25)

    return ep_ed


filterchain = TakagiSanSanFiltering()
filterchain.no_rescale_ranges = [(502, 1977)]


grain = filterchain.filtering([get_ncop(ncop_avg, clip) for clip in sources])

if __name__ == '__main__':
    encoder = X265('settings/x265_settings')

    fileinfo.name_clip_output = fileinfo.workdir / 'takagi_san_NCED01.265'
    encoder.run_enc(grain, fileinfo)
else:
    stg.output(amazon_cbr)
    stg.output(nced_sources[0])
    stg.output(get_ncop(ncop_avg, amazon_cbr))
    stg.output(grain)
