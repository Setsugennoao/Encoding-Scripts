from typing import List, Union

import vapoursynth as vs
from vardautomation import (
    JAPANESE, EztrimCutter, AudioStream, BasicTool, Mux,
    RunnerConfig, SelfRunner, VideoStream, X265Encoder,
    ChapterStream, FileInfo, Patch, QAACEncoder
)
from lvsfunc.types import Range

core = vs.core


class Encoding:
  runner: SelfRunner
  xml_tag: str = 'xml_tag.xml'
  do_chaptering: bool

  def __init__(self, file: FileInfo, clip: vs.VideoNode) -> None:
    self.file = file
    self.clip = clip

    assert self.file.a_src

    self.v_encoder = X265Encoder('mushoku_commons/x265_settings')

    self.a_extracters = [
        BasicTool('eac3to', [self.file.path.to_str(), '2:', self.file.a_src.format(track_number=1).to_str(), '-log=NUL'])
    ]

    self.a_cutters = [EztrimCutter(self.file, track=1)]
    self.a_encoders = [QAACEncoder(self.file, track=1, xml_tag=self.xml_tag)]

  def run(self, *, do_chaptering: bool = True) -> None:
    assert self.file.a_enc_cut
    self.do_chaptering = do_chaptering

    muxer = Mux(
        self.file,
        streams=(
            VideoStream(self.file.name_clip_output, 'Encoded by Setsugen no ao', JAPANESE),
            [AudioStream(self.file.a_enc_cut.format(1), 'AAC 2.0', JAPANESE)],
            ChapterStream(self.file.chapter, JAPANESE) if do_chaptering and self.file.chapter else None
        )
    )

    config = RunnerConfig(
        self.v_encoder, None,
        self.a_extracters, self.a_cutters, self.a_encoders,
        muxer
    )

    self.runner = SelfRunner(self.clip, self.file, config)
    self.runner.run()

  def do_patch(self, ranges: Union[Range, List[Range]]) -> None:
    p = Patch(self.v_encoder, self.clip, self.file, ranges)
    p.run()
    p.do_cleanup()

  def cleanup(self) -> None:
    files = [self.xml_tag]
    if self.do_chaptering:
      assert self.file.chapter
      files.append(self.file.chapter)

    self.runner.do_cleanup(*files)
