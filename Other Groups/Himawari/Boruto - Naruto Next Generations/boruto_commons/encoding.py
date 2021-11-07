import vapoursynth as vs
from typing import Optional
from .utils import get_final_filename
from vardautomation import RunnerConfig, SelfRunner, X264Encoder, FileInfo

core = vs.core


class Encoding:
  def __init__(self, file: FileInfo, clip: vs.VideoNode, prefetch: Optional[int] = None) -> None:
    self.file = file
    self.clip = clip
    self.v_encoder = X264Encoder('boruto_commons/x264_settings')
    if prefetch:
      self.v_encoder.prefetch = prefetch

  def run(self, *, do_chaptering: bool = True) -> None:
    config = RunnerConfig(self.v_encoder, None, None, None, None, None)

    runner = SelfRunner(self.clip, self.file, config)

    runner.rename_final_file(get_final_filename(self.file.path))
    runner.run()
