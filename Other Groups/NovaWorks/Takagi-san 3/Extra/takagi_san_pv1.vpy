import stgfunc as stg
import EoEfunc as eoe
import lvsfunc as lvf
import havsfunc as haf
import kagefunc as kgf
import vapoursynth as vs
from functools import partial
from vardefunc.misc import merge_chroma
from vardefunc.noise import Graigasm, AddGrain
from vsutil import get_y, iterate, depth, get_w
from debandshit import dumb3kdb, placebo_deband
from stgfunc.fades import fade_in, QuadEaseOut, QuinticEaseOut
from vardautomation import RunnerConfig, SelfRunner, X265Encoder, FileInfo, PresetWEB

core = vs.core


src_AV1 = stg.src(r".\Source\TVアニメ『からかい上手の高木さん３』PV第1弾-VrLbU-vYUDQ_AV1.mp4", 16)
src_AVC = stg.src(r".\Source\TVアニメ『からかい上手の高木さん３』PV第1弾-VrLbU-vYUDQ_AVC.mp4", 16)
src_VP9 = stg.src(r".\Source\TVアニメ『からかい上手の高木さん３』PV第1弾-VrLbU-vYUDQ_VP9.webm", 16)

fileinfo = FileInfo(r".\Source\TVアニメ『からかい上手の高木さん３』PV第1弾-VrLbU-vYUDQ_AV1.mp4", idx=stg.src, preset=[PresetWEB])

src_BD = stg.src(
    r"E:\Downloads\Torrent\...Scaricando\[BDMV] Karakai Jouzu no Takagi-san 2\TAKAGISAN2_6\BDMV\STREAM\00004.m2ts", 16
)

src_BD = src_BD[:24] + src_BD

rescaled = lvf.scale.descale(src_BD, height=range(764, 769), kernel=lvf.kernels.Robidoux())
rescaled = depth(rescaled, 16)

rescaled = eoe.dn.BM3D(rescaled, 3.75)

den = eoe.dn.BM3D(src_AVC, 5)
den_y = get_y(den)

rep = core.std.Expr([
    den_y, den_y.rgsf.Repair(den_y, 10),
    den_y.rgsf.Repair(den_y, 20)
], 'x dup + z - x y min max x y max min')

rep = haf.FastLineDarkenMOD(rep, 28, 3, 190, 4, 1)

csharp = merge_chroma(rep, den)

av1_den = src_AV1.dfttest.DFTTest(0, 4, 7, tbsize=5)

mean_average = core.average.Mean([
    av1_den, csharp, haf.SMDegrain(src_VP9, 1, 150, 50, True, True, prefilter=3)
])

merge = core.std.MaskedMerge(av1_den, mean_average, kgf.kirsch(get_y(av1_den)))

merge = lvf.rfs(merge, core.average.Mean([merge[1340], merge[1391]]) * merge.num_frames, (1340, None))

deband = core.average.Mean([
    haf.FastLineDarkenMOD(
        merge, 30, 3, 190, 4, 1
    ), merge,
    dumb3kdb(merge, 8, 18),
    dumb3kdb(merge, 12, 24),
    dumb3kdb(merge, 18, 30),
    dumb3kdb(merge, 24, 34),
    placebo_deband(merge, 8),
    placebo_deband(merge, 12),
    placebo_deband(merge, 16)
])

merge_sources = rescaled[29107:29148 + 1] + rescaled[29166:29190 + 1] + rescaled[29281:29323 + 1]

pan = src_BD[30138] * 75

pixels_y = 54
pixels_x = get_w(54)

to_zoom = pan.resize.Bicubic(
  src_BD.width * 2 + pixels_x * 2,
  src_BD.height * 2 + pixels_y * 2,
  format=vs.YUV444P16
)

x_crop = QuadEaseOut(0, pixels_x, pan.num_frames)
y_crop = QuadEaseOut(0, pixels_y, pan.num_frames)
bott_crop = QuinticEaseOut(0, 2, pan.num_frames)


def _zoom(n: int, f: vs.VideoFrame, clip: vs.VideoNode) -> vs.VideoNode:
  return clip.std.Crop(
      x_crop(n), x_crop(n), y_crop(n) * 2 - bott_crop(n), bott_crop(n)
  ).resize.Bicubic(
      src_BD.width, src_BD.height, format=src_BD.format.id
  )


merge_sources += fade_in(pan.std.FrameEval(partial(_zoom, clip=to_zoom), pan), 0, 37, QuadEaseOut)

merge_sources += rescaled[30303:30379 + 1]
merge_sources += rescaled[30419:30468 + 1]
merge_sources += rescaled[30490:30561 + 1]
merge_sources += rescaled[30705:30812 + 1]
merge_sources += rescaled.std.BlankClip(length=76)
merge_sources += rescaled.std.BlankClip(length=2).std.Invert()
merge_sources += deband[568 + 2:]

pref = iterate(get_y(merge_sources), core.std.Maximum, 2).std.Convolution([1] * 9)

grain = Graigasm(
    thrs=[x << 8 for x in (58, 80, 130)],
    strengths=[(0.7, 0.08), (0.4, 0.06), (0.2, 0)],
    sizes=(1.24, 1.16, 1.10),
    sharps=(65, 45, 40),
    grainers=[
        AddGrain(seed=69 * 420, constant=True),
        AddGrain(seed=420 * 69, constant=True),
        AddGrain(seed=21 * 365, constant=True)
    ]
).graining(merge_sources, prefilter=pref)

grain = depth(grain, 10).std.Limiter(16 << 2, [235 << 2, 240 << 2], [0, 1, 2])

if __name__ == "__main__":
    encoder = X265Encoder("x265_settings")
    encoder.prefetch = 1

    fileinfo.name_clip_output = fileinfo.workdir / 'takagi_san_3_pv01.265'

    SelfRunner(grain, fileinfo, RunnerConfig(encoder)).run()
else:
  stg.output(src_AV1)
  stg.output(grain)
