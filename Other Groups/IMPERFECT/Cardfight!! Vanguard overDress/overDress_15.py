import stgfunc as stg
import lvsfunc as lvf
import vapoursynth as vs
from vardautomation import FileInfo, PresetAAC, PresetWEB

from overdress_commons import overDressFiltering, Encoding

core = vs.core

CR = FileInfo(r".\Source\[SubsPlease] Cardfight!! Vanguard overDress - 15 (1080p) [45695751].mkv", preset=(PresetWEB, PresetAAC))


class Filtering(overDressFiltering):
  opening_ranges = (1344, 3500)
  ending_ranges = (31768, 33926)
  oshirase_ranges = (34527, 35244)

  def custom(self, src, den, lines, unsharp, deband, grain):
    return lvf.rfs(grain, lines, [(35245, 35292), (35503, None)])


chain = Filtering(CR)
filtered = chain.filterchain()

if __name__ == '__main__':
  brrrr = Encoding(CR, filtered)
  brrrr.run()
else:
  stg.output(CR.clip_cut)
  stg.output(filtered)
