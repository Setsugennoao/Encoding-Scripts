import stgfunc as stg
import vapoursynth as vs
from vardautomation import X265, FileInfo, PresetWEB, get_vs_core

from takagi_san_commons.filtering import TakagiSanSanFiltering

core = get_vs_core(range(0, vs.core.num_threads, 2))

fileinfo = FileInfo(
    r".\Source\04\Karakai Jouzu no Takagi-san S3 - 04 (Amazon dAnime VBR 1080p).mkv", preset=[PresetWEB]
)

sources = [
    stg.src(
        r".\Source\04\Karakai Jouzu no Takagi-san S3 - 04 (Amazon dAnime CBR 1080p).mkv", 16, matrix_prop=1,
        force_lsmas=True),
    stg.src(
        r".\Source\04\Karakai Jouzu no Takagi-san S3 - 04 (Amazon dAnime VBR 1080p).mkv", 16, matrix_prop=1,
        force_lsmas=True),
    stg.src(
        r".\Source\04\Karakai Jouzu no Takagi-san S3 - 04 (B-Global CBR 1080p).mkv", 16, matrix_prop=1,
        force_lsmas=True),
    stg.src(
        r".\Source\04\Karakai Jouzu no Takagi-san S3 - 04 (Netflix VBR 1080p).mkv", 16, matrix_prop=1, force_lsmas=True)
    [24: -24]]

filterchain = TakagiSanSanFiltering()

grain = filterchain.filtering(sources)

if __name__ == '__main__':
    encoder = X265('settings/x265_settings')

    ranges = [(2158, 31838)]

    for i, (start, end) in enumerate(ranges):
        fileinfo.name_clip_output = fileinfo.workdir / f'takagi_san_04v1_part-{i}-{len(ranges)}.265'
        encoder.run_enc(grain[slice(start, end + 1)], fileinfo)
else:
    # stg.output(lvf.diff(sources[0], sources[2]))
    stg.output(sources[0])
    stg.output(grain)
