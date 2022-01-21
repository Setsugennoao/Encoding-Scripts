import vapoursynth as vs
from lvsfunc.types import Range
from typing import List, Union, Optional
from vardautomation import (
    JAPANESE, AudioStream, Mux, RunnerConfig, SelfRunner,
    VideoStream, X265, FileInfo, Patch
)

core = vs.core


class Encoding:
    runner: SelfRunner
    xml_tag: str = 'xml_tag.xml'
    do_chaptering: bool

    def __init__(self, file: FileInfo, clip: vs.VideoNode, prefetch: Optional[int] = None) -> None:
        self.file = file
        self.clip = clip
        self.v_encoder = X265('selection_project_commons/x265_settings')
        self.file.set_name_clip_output_ext('.265')
        if prefetch:
            self.v_encoder.prefetch = prefetch

    def run(self, *, do_chaptering: bool = True) -> None:
        assert self.file.a_src, self.file.a_src_cut

        muxer = Mux(
            self.file,
            streams=(
                VideoStream(self.file.name_clip_output, 'Encoded by Setsugen no ao', JAPANESE),
                AudioStream(self.file.a_enc_cut.format(1), 'Funimation Stereo AAC', JAPANESE),
                None
            )
        )

        config = RunnerConfig(self.v_encoder, None, None, None, None, muxer)

        runner = SelfRunner(self.clip, self.file, config)
        runner.run()
        runner.cleanup_files.remove(self.file.name_clip_output)
        runner.do_cleanup(self.file.a_src.set_track(1), self.file.a_src_cut.set_track(1))

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
