import vapoursynth as vs
from typing import Optional
from .utils import get_final_filename
from vardautomation import X264, FileInfo

core = vs.core


class Encoding:
    def __init__(self, file: FileInfo, clip: vs.VideoNode, prefetch: Optional[int] = None) -> None:
        self.file = file
        self.clip = clip
        self.v_encoder = X264('boruto_commons/x264_settings')
        self.v_encoder.resumable = True

        if prefetch:
            self.v_encoder.prefetch = prefetch

    def run(self, *, do_chaptering: bool = True) -> None:
        self.file.name_clip_output = get_final_filename(self.file.path)

        self.v_encoder.run_enc(
            self.clip, self.file, self.file.clip_cut.resize.Bilinear(1280, 720)
        )
