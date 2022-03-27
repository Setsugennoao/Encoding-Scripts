from vsdpir import DPIR
import vapoursynth as vs

core = vs.core


class Deblocking:
    def __init__(self, is_preview: bool = False):
        self.is_preview = is_preview
        self.DPIR_ARGS = dict(
            task='deblock', fp16=True, device_type='cuda', device_index=0
        )

    def deblock_DPIR(self, clip: vs.VideoNode, strength: int, **DPIR_kwargs) -> vs.VideoNode:
        if self.is_preview:
            self.DPIR_ARGS |= dict(
                trt=True, tile_x=960, tile_y=540, tile_pad=16,
            )

        return DPIR(
            core.resize.Bicubic(
                clip.grain.Add(strength / 95, strength / 750), format=vs.RGBS, matrix_in=1
            ), strength, **self.DPIR_ARGS, **DPIR_kwargs
        )
