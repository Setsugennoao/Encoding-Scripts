import stgfunc as stg
import lvsfunc as lvf
import vapoursynth as vs
from vsutil import get_y
from lvsfunc.kernels import Catrom
from stgfunc.transitions import fade
from stgfunc.utils import replace_squaremask
from vardautomation import X265, FileInfo, PresetWEB, get_vs_core

from takagi_san_commons.filtering import TakagiSanSanFiltering

core = get_vs_core(range(0, vs.core.num_threads, 2))
catrom = Catrom()

fileinfo = FileInfo(
    r".\Source\02\Karakai Jouzu no Takagi-san S3 - 02 (Amazon dAnime VBR 1080p).mkv", preset=[PresetWEB]
)

sources = [
    stg.src(r".\Source\02\Karakai Jouzu no Takagi-san S3 - 02 (Amazon dAnime CBR 1080p).mkv", 16, matrix_prop=1),
    stg.src(r".\Source\02\Karakai Jouzu no Takagi-san S3 - 02 (Amazon dAnime VBR 1080p).mkv", 16, matrix_prop=1),
    stg.src(r".\Source\02\Karakai Jouzu no Takagi-san S3 - 02 (Netflix VBR 1080p).mkv", 16, matrix_prop=1)[24:-24]
]

sources = [source[31838:] for source in sources]
sources = [*sources[:2], core.average.Mean(sources), sources[2]]

nced_sources = [
    stg.src(
        r".\Source\NCED02\TVアニメ『からかい上手の高木さん』ノンクレジットED「Over Drive」／高木さん(CV -高橋李依)-udXPAhWqi08.webm", 16, matrix_prop=1)
    [1:],
    stg.src(
        r".\Source\NCED02\TVアニメ『からかい上手の高木さん』ノンクレジットED「Over Drive」／高木さん(CV -高橋李依)-udXPAhWqi08.mp4", 16, matrix_prop=1)
    [1:]]

amazon_cbr = sources[1]
ncop_avg = core.average.Mean(nced_sources)

black_gray = get_y(amazon_cbr).std.BlankClip()
white_gray = black_gray.std.Invert()


def get_ncop(nced: vs.VideoNode, ep_ed: vs.VideoNode) -> vs.VideoNode:
    green = core.std.BlankClip(ep_ed, color=[54528, 26624, 29184])

    ep_ed = replace_squaremask(ep_ed, nced, (411, 395, 759, 343), (71, 178))
    ep_ed = replace_squaremask(ep_ed, nced, (451, 355, 753, 345), (185, 288))

    ed_static = core.average.Mean([ep_ed[f] for f in [
        *range(445, 450 + 1), *range(571, 576 + 1), *range(697, 702 + 1),
        *range(823, 828 + 1), *range(1879, 1884 + 1), *range(1955, 1960 + 1),

    ]]) * ep_ed.num_frames

    # Ending dehardsubbing
    ep_ed = replace_squaremask(ep_ed, green, (1165, 1080, 0, 0), [(967, 1220), (1570, 1878)])
    ep_ed = replace_squaremask(ep_ed, green, (1155, 1080, 765, 0), [(1239, 1551)])

    ep_ed = replace_squaremask(ep_ed, ed_static, (435, 900, 80, 100), (325, 948), 0.4)
    ep_ed = replace_squaremask(ep_ed, ed_static, (720, 310, 500, 770), (325, 948), 0.4)
    ep_ed = replace_squaremask(ep_ed, ed_static, (645, 180, 520, 580), (325, 948), 0.4)

    ep_ed = replace_squaremask(ep_ed, ed_static, (190, 240, 530, 547), (451, 570), 0.4)
    ep_ed = replace_squaremask(ep_ed, ed_static, (430, 63, 500, 530), (577, 696), 0.4)
    ep_ed = replace_squaremask(ep_ed, ed_static, (217, 320, 497, 467), (703, 822), 0.4)

    ep_ed = replace_squaremask(ep_ed, ed_static, (425, 135, 520, 430), (829, 948), 0.4)

    ep_ed = replace_squaremask(ep_ed, ed_static, (600, 435, 230, 325), (1885, 2030), 0.4)

    ep_ed = replace_squaremask(ep_ed, ed_static, (211, 65, 740, 535), (703, 822), 0.4)

    ep_ed = replace_squaremask(ep_ed, nced, (230, 120, 975, 490), (577, 948), 0.5)

    ep_ed = replace_squaremask(ep_ed, nced, (83, 38, 1094, 735), (2044, None), 0.25)
    ep_ed = replace_squaremask(ep_ed, nced, (197, 42, 1231, 733), (2044, None), 0.25)
    ep_ed = replace_squaremask(ep_ed, nced, (243, 73, 1452, 699), (2044, None), 0.25)
    ep_ed = replace_squaremask(ep_ed, nced, (83, 41, 1096, 830), (2044, None), 0.25)
    ep_ed = replace_squaremask(ep_ed, nced, (553, 44, 1227, 828), (2044, None), 0.25)
    ep_ed = replace_squaremask(ep_ed, nced, (770, 38, 1042, 981), (2044, None), 0.25)

    # Redoing the same fade that is present in ED01. The direct cut looks like shit
    ep_ed = fade(
        lvf.rfs(ep_ed, ep_ed[1878] * ep_ed.num_frames, (1879, 1887)),
        lvf.rfs(ep_ed, ep_ed[1879] * ep_ed.num_frames, (1869, 1887)),
        True, 1869, 1888
    )

    return ep_ed


filterchain = TakagiSanSanFiltering()
filterchain.no_rescale_ranges = [(956, 1883)]


grain = filterchain.filtering([get_ncop(ncop_avg, clip) for clip in sources])

if __name__ == '__main__':
    encoder = X265('settings/x265_settings')

    fileinfo.name_clip_output = fileinfo.workdir / 'takagi_san_NCED02.265'
    encoder.run_enc(grain, fileinfo)
else:
    stg.output(amazon_cbr)
    stg.output(nced_sources[0])
    stg.output(get_ncop(ncop_avg, amazon_cbr))
    stg.output(grain)
