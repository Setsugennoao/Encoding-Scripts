import stgfunc as stg
import vapoursynth as vs
from pathlib import Path
from vardautomation import FileInfo, PresetOpus, get_vs_core
from kyounoaquaa_commons import Encoding, KyouNoAquaaFiltering

core = get_vs_core(range(0, vs.core.num_threads, 2))

YOUTUBE = FileInfo(r".\Source\『きょうのAqours』第2回　渡辺 曜-E8G5HMS27QQ_AV1.mp4", idx=stg.src, preset=(PresetOpus))
YOUTUBE.a_src = Path(r".\Source\『きょうのAqours』第2回　渡辺 曜-E8G5HMS27QQ_OPUS.webm")


class Filtering(KyouNoAquaaFiltering):
  pass


chain = Filtering(YOUTUBE)
filtered = chain.filterchain()

if __name__ == '__main__':
  brrrr = Encoding(YOUTUBE, filtered)
  brrrr.run()
else:
  stg.output(YOUTUBE.clip_cut)
  stg.output(filtered)
