import stgfunc as stg
import vapoursynth as vs
from vardautomation import FileInfo, PresetAAC, PresetWEB

from overdress_commons import overDressFiltering, Encoding

core = vs.core

CR = FileInfo(r".\Source\[SubsPlease] Cardfight!! Vanguard overDress - 13 (1080p) [7600F98A].mkv", preset=(PresetWEB, PresetAAC))


class Filtering(overDressFiltering):
  ending_only_credits = True
  ending_ranges = (30200, 32367)
  oshirase_ranges = (32368, 33087)


chain = Filtering(CR)
filtered = chain.filterchain()

if __name__ == '__main__':
  brrrr = Encoding(CR, filtered)
  brrrr.run()
else:
  stg.output(CR.clip_cut)
  stg.output(filtered)
