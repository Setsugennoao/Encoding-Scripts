import stgfunc as stg
from vardautomation import FileInfo, PresetAAC, PresetWEB

from mushoku_commons import MushokuFiltering, Encoding

FUNI = FileInfo(r".\Source\[SubsPlease] Mushoku Tensei - 21 (1080p) [BE2036FD].mkv", (168, 0), preset=[PresetWEB, PresetAAC])


class Filtering(MushokuFiltering):
  denoise_strength = [None, 0.30]


chain = Filtering(FUNI)
filtered = chain.filterchain()

if __name__ == '__main__':
  brrrr = Encoding(FUNI, filtered)
  brrrr.run()
else:
  stg.output(FUNI.clip_cut)
  stg.output(filtered)
