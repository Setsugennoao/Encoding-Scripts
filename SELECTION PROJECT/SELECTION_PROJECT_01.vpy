import stgfunc as stg
from stgfunc.utils import replace_squaremask
from vardautomation import FileInfo, PresetAAC, PresetWEB
from selection_project_commons import SeleProFiltering, Encoding


FUNI = FileInfo(r".\Source\[SubsPlease] Selection Project - 01 (1080p) [47D622BA].mkv", (240, 0), preset=(PresetWEB, PresetAAC))
BILI = FileInfo(r".\Source\[NC-Raws] SELECTION PROJECT - 01 [B-Global][WEB-DL][1080p][AVC AAC][Multiple Subtitle][MKV].mkv", preset=(PresetWEB))


class Filtering(SeleProFiltering):
  def mix_sources(self):
    # MUSE Asia signs
    self.BILI.clip_cut = replace_squaremask(self.BILI.clip_cut, self.FUNI.clip_cut, (360, 150, 0, 930), [(0, 74), (33731, 33807)])

    # Lol they fucked up the name and placement
    self.BILI.clip_cut = replace_squaremask(self.BILI.clip_cut, self.FUNI.clip_cut, (210, 110, 550, 660), (32763, 32856))
    self.FUNI.clip_cut = replace_squaremask(self.FUNI.clip_cut, self.BILI.clip_cut, (200, 50, 300, 665), (32763, 32856))

    # stg.output(lvf.diff(self.FUNI.clip_cut, self.BILI.clip_cut))


chain = Filtering(FUNI, BILI)
filtered = chain.filterchain()

if __name__ == '__main__':
  brrrr = Encoding(FUNI, filtered)
  brrrr.run()
else:
  stg.output(FUNI.clip_cut)
  stg.output(BILI.clip_cut)
  stg.output(filtered)
