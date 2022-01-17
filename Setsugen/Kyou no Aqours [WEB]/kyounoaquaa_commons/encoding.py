import vapoursynth as vs
from typing import Optional
from vardautomation import (
    JAPANESE, AudioStream, Mux, RunnerConfig,
    SelfRunner, VideoStream, X265, FileInfo
)

core = vs.core


class Encoding:
  runner: SelfRunner

  def __init__(self, file: FileInfo, clip: vs.VideoNode, prefetch: Optional[int] = None) -> None:
    self.file = file
    self.clip = clip
    self.v_encoder = X265('kyounoaquaa_commons/x265_settings')
    if prefetch:
      self.v_encoder.prefetch = prefetch

  def run(self,) -> None:
    assert self.file.a_src, self.file.a_src_cut

    muxer = Mux(
        self.file,
        streams=(
            VideoStream(self.file.name_clip_output, 'Encoded by Setsugen no ao', JAPANESE),
            AudioStream(self.file.a_src, 'YouTube Stereo OPUS', JAPANESE),
            None
        )
    )

    config = RunnerConfig(self.v_encoder, None, None, None, None, muxer)

    SelfRunner(self.clip, self.file, config).run()
