import stgfunc as stg
from vardautomation import FileInfo
from vsencode import EncodeRunner
from sumtimerender_commons import filterchain, get_encoder, EPS_ED_RANGES

episode = 11

fileinfo = FileInfo(fr".\Source\{episode}\Summer Time Rendering {episode} - (Disney+).mkv", (24, -24))
src = fileinfo.clip_cut

ED_RANGES = EPS_ED_RANGES[episode - 1]

MEDIUM_GRAIN_BUT_IDK_MAN_THE_MOTION_BLOCKS_ARE_DYING = [
]

SUPER_COARSE_GRAINY_WTF_KILL_THIS_STUDIO_PLEASE_RANGES = [
]

VSDPIR_DEBLOCK_RANGES_JESUSSSSS = [
]

EPIC_DEBANDING_RANGES = [
    (31181, 31527)
]

grain = filterchain(
    episode - 1,
    MEDIUM_GRAIN_BUT_IDK_MAN_THE_MOTION_BLOCKS_ARE_DYING,
    SUPER_COARSE_GRAINY_WTF_KILL_THIS_STUDIO_PLEASE_RANGES,
    VSDPIR_DEBLOCK_RANGES_JESUSSSSS,
    EPIC_DEBANDING_RANGES
)

if __name__ == '__main__':
    fileinfo.name_clip_output = fileinfo.workdir / 'smtr_11v1.265'

    # runner = EncodeRunner(fileinfo, grain)

    get_encoder(
        ED_RANGES,
        MEDIUM_GRAIN_BUT_IDK_MAN_THE_MOTION_BLOCKS_ARE_DYING,
        SUPER_COARSE_GRAINY_WTF_KILL_THIS_STUDIO_PLEASE_RANGES
    ).run_enc(grain, fileinfo)

    # runner.video(encoder, qp_clip=src)
    # runner.audio('qaac')
    # runner.run()
else:
    stg.output(src, False)
    stg.output(grain, False)
