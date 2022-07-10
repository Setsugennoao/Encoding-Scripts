import stgfunc as stg
from vardautomation import FileInfo
from sumtimerender_commons import filterchain, get_encoder, EPS_ED_RANGES

cour = 2
episode = 13

fileinfo = FileInfo(fr".\Source\{episode}\Summer Time Rendering {episode} - (Disney+).mkv", (96, -24))
src = fileinfo.clip_cut

ED_RANGES = EPS_ED_RANGES[cour - 1][episode - 12 - 1]

MEDIUM_GRAIN_BUT_IDK_MAN_THE_MOTION_BLOCKS_ARE_DYING = [
    (2552, 2783)
]

SUPER_COARSE_GRAINY_WTF_KILL_THIS_STUDIO_PLEASE_RANGES = [
    (5027, 5265), (11347, 12921), (16896, 17078),  # schizo studio STOP PLEASE
    (18818, 18923)
]

VSDPIR_DEBLOCK_RANGES_JESUSSSSS = [
    (20679, 20854), (20963, 22297)
]

BIG_ASS_GRAIN_DUDE_PLEASE_CMON = [
    (22316, 31528)
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
    fileinfo.name_clip_output = fileinfo.workdir / 'smtr_13v1.265'

    get_encoder(
        ED_RANGES,
        MEDIUM_GRAIN_BUT_IDK_MAN_THE_MOTION_BLOCKS_ARE_DYING,
        SUPER_COARSE_GRAINY_WTF_KILL_THIS_STUDIO_PLEASE_RANGES
    ).run_enc(grain, fileinfo)
else:
    stg.output(src, False)
    stg.output(grain, False)
