import stgfunc as stg
from vardautomation import FileInfo
from sumtimerender_commons import filterchain, get_encoder, EPS_ED_RANGES

episode = 10

fileinfo = FileInfo(fr".\Source\{episode}\Summer Time Rendering {episode} - (Disney+).mkv", (24, -24))
src = fileinfo.clip_cut

ED_RANGES = EPS_ED_RANGES[episode - 1]

MEDIUM_GRAIN_BUT_IDK_MAN_THE_MOTION_BLOCKS_ARE_DYING = [
    (3061, 3271), (7413, 7716), (28242, 28332)
]

SUPER_COARSE_GRAINY_WTF_KILL_THIS_STUDIO_PLEASE_RANGES = [
    (27330, 27435), (30252, 30391)
]

VSDPIR_DEBLOCK_RANGES_JESUSSSSS = [
    (27330, 27435), (28242, 28332), (30252, 30391)  # schizo studio
]

EPIC_DEBANDING_RANGES = [
    (18129, 18175),
    (18223, 18293),
    (22410, 22456),
    (24061, 24131),
    (24922, 25009),
    (25402, 25556)
]

grain = filterchain(
    episode - 1,
    MEDIUM_GRAIN_BUT_IDK_MAN_THE_MOTION_BLOCKS_ARE_DYING,
    SUPER_COARSE_GRAINY_WTF_KILL_THIS_STUDIO_PLEASE_RANGES,
    VSDPIR_DEBLOCK_RANGES_JESUSSSSS,
    EPIC_DEBANDING_RANGES
)

if __name__ == '__main__':
    fileinfo.name_clip_output = fileinfo.workdir / 'smtr_10v1.265'

    get_encoder(
        ED_RANGES,
        MEDIUM_GRAIN_BUT_IDK_MAN_THE_MOTION_BLOCKS_ARE_DYING,
        SUPER_COARSE_GRAINY_WTF_KILL_THIS_STUDIO_PLEASE_RANGES
    ).run_enc(grain, fileinfo, qpfile_clip=src)
else:
    stg.output(src, False)
    stg.output(grain, False)
