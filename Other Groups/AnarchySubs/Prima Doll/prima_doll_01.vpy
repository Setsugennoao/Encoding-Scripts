import vapoursynth as vs
from debandshit import dumb3kdb
from stgfunc import set_output, source
from vardefunc import finalise_clip
from vsdehalo import edge_cleaner, fine_dehalo
from vsdenoise import BM3DCudaRTC, CCDMode, CCDPoints, ccd
from vsencode import EncodeRunner, FileInfo
from vskernels import Catrom
from vsmask.edge import FDoG
from vsrgtools import contrasharpening_dehalo
from vsutil import depth, get_y

core = vs.core
catrom = Catrom()

fileinfo = FileInfo(r".\Source\Prima Doll - 01 (Amazon dAnime CBR 1080p).mkv")

clip_cbr = depth(fileinfo.clip_cut, 16)

clip_vbr = source(r".\Source\Prima Doll - 01 (Amazon dAnime VBR 1080p).mkv", 16)

average_clips = [clip_cbr, clip_vbr]
average_clips += [clip_cbr.deblock.Deblock(quant, 2, planes=0) for quant in {4, 6, 8, 16, 24}]
average_clips += [clip_vbr.deblock.Deblock(quant, 4, planes=0) for quant in {8, 16, 18}]

average = core.average.Mean(average_clips)

denoise = BM3DCudaRTC(average, [1.25, 0], 0).clip

deband = dumb3kdb(denoise)
deband = contrasharpening_dehalo(deband, average, 1.6)

cdenoise = ccd(
    deband, 3.5, mode=CCDMode.BICUBIC_LUMA, ref_points=CCDPoints.MEDIUM, scale=1.5
)

dehalo = fine_dehalo(cdenoise, None, 2.1, 2.1, 0, 1.2, 80, 100)

clean = edge_cleaner(dehalo, 5.75, hot=True, smode=True)

edgemask = FDoG().edgemask(get_y(clean))

aa = clean.std.MaskedMerge(clean.eedi2cuda.AA2(0, 0, 0, 0,  planes=0), edgemask)

grain = aa.noise.Add(0.075)

grain = finalise_clip(grain)


if __name__ == '__main__':
    fileinfo.name_clip_output = fileinfo.workdir / 'prima_doll_01.265'

    EncodeRunner(fileinfo, grain) \
        .video('x265', 'prima_doll_x265_settings', qp_clip=False) \
        .audio('passthrough') \
        .mux() \
        .run()
else:
    set_output(clip_cbr)
    set_output(grain)
