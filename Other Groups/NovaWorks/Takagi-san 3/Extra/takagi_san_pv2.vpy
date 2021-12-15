import stgfunc as stg
import EoEfunc as eoe
import lvsfunc as lvf
import havsfunc as haf
import vapoursynth as vs
from vsutil import get_y, iterate, depth
from vardefunc.noise import Graigasm, AddGrain
from debandshit import dumb3kdb, placebo_deband
from vardautomation import RunnerConfig, SelfRunner, X265Encoder, FileInfo, PresetWEB

core = vs.core

fileinfo = FileInfo(r".\Source\TVアニメ『からかい上手の高木さん３』PV第2弾（2022年1月7日（金）放送開始！）-N238ads-bdw_AV1.mp4", idx=stg.src, preset=[PresetWEB])

src_AV1_PV01 = stg.src(r".\Source\TVアニメ『からかい上手の高木さん３』PV第1弾-VrLbU-vYUDQ_AV1.mp4", 16)
oblique_lines = core.average.Mean([src_AV1_PV01[1340], src_AV1_PV01[1391]])

src_AV1 = depth(fileinfo.clip_cut, 16) + oblique_lines
src_AVC = stg.src(r".\Source\TVアニメ『からかい上手の高木さん３』PV第2弾（2022年1月7日（金）放送開始！）-N238ads-bdw_AVC.mp4", 16) + oblique_lines
src_VP9 = stg.src(r".\Source\TVアニメ『からかい上手の高木さん３』PV第2弾（2022年1月7日（金）放送開始！）-N238ads-bdw_VP9.webm", 16) + oblique_lines

den = core.average.Mean([eoe.dn.BM3D(src_AVC, 3), src_AV1.cas.CAS(1)])

av1_den = src_AV1.dfttest.DFTTest(0, 4, 7, tbsize=5)

mean = haf.SMDegrain(core.average.Mean([av1_den, den]), 1, 250, 75, True, True, prefilter=3)

mean_ec = haf.EdgeCleaner(mean, 18)

merge = lvf.rfs(mean_ec, mean[-1] * mean.num_frames, (660, 677))[:-1]

merge = lvf.aa.based_aa(merge, stg.upscale._SHADERS).cas.CAS(0.24)

deband = core.average.Mean([
    haf.FastLineDarkenMOD(
        merge, -20, 3, 190, 4, 1
    ), merge,
    dumb3kdb(merge, 8, 18),
    dumb3kdb(merge, 12, 24),
    dumb3kdb(merge, 18, 30),
    dumb3kdb(merge, 24, 34),
    placebo_deband(merge, 8),
    placebo_deband(merge, 12),
    placebo_deband(merge, 16)
])

pref = iterate(get_y(deband), core.std.Maximum, 2).std.Convolution([1] * 9)

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
).graining(deband, prefilter=pref)

grain = depth(grain, 10).std.Limiter(16 << 2, [235 << 2, 240 << 2], [0, 1, 2])

if __name__ == "__main__":
  encoder = X265Encoder("x265_settings")
  encoder.prefetch = 1

  fileinfo.name_clip_output = fileinfo.workdir / 'takagi_san_3_pv02.265'

  SelfRunner(grain, fileinfo, RunnerConfig(encoder)).run()
else:
  stg.output(src_AV1)
  stg.output(grain)
