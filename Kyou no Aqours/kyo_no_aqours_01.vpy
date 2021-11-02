import stgfunc as stg
import vapoursynth as vs
from pathlib import Path
from vsutil import insert_clip
from vardautomation import FileInfo, PresetOpus
from kyonoaquaa_commons import Encoding, KyoNoAquaaFiltering

core = vs.core

YOUTUBE = FileInfo(r"E:\Desktop\きょうのAqours\『きょうのAqours』第1回 高海千歌-sJqT3w813q8_AV1.mp4", idx=stg.src, preset=(PresetOpus))
YOUTUBE.a_src = Path(r"E:\Desktop\きょうのAqours\『きょうのAqours』第1回 高海千歌-sJqT3w813q8_OPUS.webm")


class Filtering(KyoNoAquaaFiltering):
  def mix_source(self):
    src = self.YOUTUBE.clip_cut
    return insert_clip(src, src[4088], 4087)


chain = Filtering(YOUTUBE)
filtered = chain.filterchain()

if __name__ == '__main__':
  brrrr = Encoding(YOUTUBE, filtered)
  brrrr.run()
else:
  stg.output(YOUTUBE.clip_cut)
  stg.output(filtered)
