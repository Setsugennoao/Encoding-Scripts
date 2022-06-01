import stgfunc as stg
from vardautomation import FileInfo

from sumtimerender_commons import filterchain, get_encoder, EPS_ED_RANGES

episode = 5

fileinfo = FileInfo(fr".\Source\0{episode}\Summer Time Rendering 0{episode} - (Disney+).mkv", (24, -24))
src = fileinfo.clip_cut

ED_RANGES = EPS_ED_RANGES[episode - 1]

MEDIUM_GRAIN_BUT_IDK_MAN_THE_MOTION_BLOCKS_ARE_DYING = [
    (22267, 22458)
]

SUPER_COARSE_GRAINY_WTF_KILL_THIS_STUDIO_PLEASE_RANGES = [
    (485, 532)
]

VSDPIR_DEBLOCK_RANGES_JESUSSSSS = [
    (26397, 27394)
]

grain = filterchain(
    episode - 1,
    MEDIUM_GRAIN_BUT_IDK_MAN_THE_MOTION_BLOCKS_ARE_DYING,
    SUPER_COARSE_GRAINY_WTF_KILL_THIS_STUDIO_PLEASE_RANGES,
    VSDPIR_DEBLOCK_RANGES_JESUSSSSS
)

if __name__ == '__main__':
    fileinfo.name_clip_output = fileinfo.workdir / 'smtr_05v1.265'

    get_encoder(
        ED_RANGES,
        MEDIUM_GRAIN_BUT_IDK_MAN_THE_MOTION_BLOCKS_ARE_DYING,
        SUPER_COARSE_GRAINY_WTF_KILL_THIS_STUDIO_PLEASE_RANGES,
        # override_params={'--crf': 16}
    ).run_enc(grain, fileinfo, qpfile_clip=src)
else:
    stg.output(src, False)
    stg.output(grain, False)
