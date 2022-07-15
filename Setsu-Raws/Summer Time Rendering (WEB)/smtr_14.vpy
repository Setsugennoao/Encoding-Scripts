import stgfunc as stg
from vardautomation import FileInfo
from sumtimerender_commons import filterchain, get_encoder, EPS_ED_RANGES

cour = 2
episode = 14

fileinfo = FileInfo(fr".\Source\{episode}\Summer Time Rendering {episode} - (Disney+).mkv", (96, -24))
src = fileinfo.clip_cut

ED_RANGES = EPS_ED_RANGES[cour - 1][episode - 12 - 1]

MEDIUM_GRAIN_BUT_IDK_MAN_THE_MOTION_BLOCKS_ARE_DYING = [
]

SUPER_COARSE_GRAINY_WTF_KILL_THIS_STUDIO_PLEASE_RANGES = [
]

VSDPIR_DEBLOCK_RANGES_JESUSSSSS = [
]

BIG_ASS_GRAIN_DUDE_PLEASE_CMON = [
]

EPIC_DEBANDING_RANGES = []

grain = filterchain(
    episode - 1,
    MEDIUM_GRAIN_BUT_IDK_MAN_THE_MOTION_BLOCKS_ARE_DYING,
    SUPER_COARSE_GRAINY_WTF_KILL_THIS_STUDIO_PLEASE_RANGES,
    VSDPIR_DEBLOCK_RANGES_JESUSSSSS,
    EPIC_DEBANDING_RANGES,
    BIG_ASS_GRAIN_DUDE_PLEASE_CMON,
    cour
)

if __name__ == '__main__':
    fileinfo.name_clip_output = fileinfo.workdir / 'smtr_14v1.265'

    get_encoder(
        ED_RANGES,
        MEDIUM_GRAIN_BUT_IDK_MAN_THE_MOTION_BLOCKS_ARE_DYING,
        SUPER_COARSE_GRAINY_WTF_KILL_THIS_STUDIO_PLEASE_RANGES
    ).run_enc(grain, fileinfo)
else:
    stg.output(src, False)
    stg.output(grain, False)
