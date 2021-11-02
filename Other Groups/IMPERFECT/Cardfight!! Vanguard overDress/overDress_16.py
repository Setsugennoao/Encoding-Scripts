import stgfunc as stg
import lvsfunc as lvf
import vapoursynth as vs
from vardautomation import FileInfo, PresetAAC, PresetWEB

from overdress_commons import overDressFiltering, Encoding

core = vs.core

CR = FileInfo(r".\Source\[SubsPlease] Cardfight!! Vanguard overDress - 16 (1080p) [A4910B6D].mkv", preset=(PresetWEB, PresetAAC))


class Filtering(overDressFiltering):
  opening_ranges = (984, 3140)
  ending_ranges = (32367, 34525)
  oshirase_ranges = (34526, 35243)

  def custom(self, src, den, lines, unsharp, deband, grain):
    return lvf.rfs(grain, lines, [(35244, 35291), (35502, None)])


chain = Filtering(CR)
filtered = chain.filterchain()

if __name__ == '__main__':
  brrrr = Encoding(CR, filtered)
  brrrr.run()
else:
  stg.output(CR.clip_cut)
  stg.output(filtered)
