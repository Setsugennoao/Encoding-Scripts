import stgfunc as stg
import vapoursynth as vs
from vardautomation import X265, FileInfo, PresetWEB

from jojo_so_commons.filtering import StoneOceanFiltering

core = vs.core


fileinfo = FileInfo(
    r"E:\Desktop\Encoding Sources\JoJo Stone Ocean\JoJo Stone Ocean - EP01v0 (01.12.21).mkv",
    (323, 34238), idx=stg.src, preset=[PresetWEB]
)

src = fileinfo.clip_cut

chain = StoneOceanFiltering(fileinfo)
chain.BLOCKY_AF_RANGES = [(16384, 16559)]

grain = chain.filterchain(src)


if __name__ == '__main__':
    encoder = X265('jojo_so_commons/x265_settings')
    encoder.prefetch = 1

    fileinfo.name_clip_output = fileinfo.workdir / 'jojo_so_01v2.265'

    encoder.run_enc(grain, fileinfo)
else:
    stg.output(src)
    stg.output(grain)

# from vardautomation.comp import make_comps
# make_comps({"Source": src, "Filtering": grain}, num=30, slowpics=True, force_bt709=True, collection_name="JoJo: Stone Ocean Encode")
