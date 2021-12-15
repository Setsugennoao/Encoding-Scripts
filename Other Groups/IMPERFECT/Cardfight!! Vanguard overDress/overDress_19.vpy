import stgfunc as stg
import lvsfunc as lvf
import vapoursynth as vs
from vardautomation import FileInfo, PresetAAC, PresetWEB

from overdress_commons import overDressFiltering, Encoding

core = vs.core

CR = FileInfo(r".\Source\[SubsPlease] Cardfight!! Vanguard overDress - 19 (1080p) [F8F0E930].mkv", preset=(PresetWEB, PresetAAC))


class Filtering(overDressFiltering):
  opening_ranges = (2134, 4292)
  ending_ranges = (32369, 34527)
  oshirase_ranges = (34528, 35245)

  def custom(self, src, den, lines, unsharp, deband, grain):
    return lvf.rfs(grain, lines, [(35246, 35294), (35504, None)])


chain = Filtering(CR)
filtered = chain.filterchain()

if __name__ == '__main__':
  brrrr = Encoding(CR, filtered)
  brrrr.run()
else:
  stg.output(CR.clip_cut, False)
  stg.output(filtered)
