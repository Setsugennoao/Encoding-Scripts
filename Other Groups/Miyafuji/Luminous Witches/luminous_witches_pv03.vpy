from ccd import ccd
import stgfunc as stg
import lvsfunc as lvf
import vapoursynth as vs
from vsmlrt import Backend
from vsutil import iterate, get_y
from vsmask.edge import FDoGTCanny
from lvsfunc.deblock import vsdpir
from vardautomation import X265, FileInfo
from vardefunc.noise import Graigasm, AddGrain
from vardefunc.util import initialise_clip, finalise_clip
from debandshit import dumb3kdb, placebo_deband, f3kbilateral
from fine_dehalo import contrasharpening, contrasharpening_fine_dehalo as contra_fdha


core = vs.core


src_dir = r"E:\Desktop\Encoding Sources\Strike_Witches\Luminous Witches\PV3"

fileinfo = FileInfo(rf"{src_dir}\TVアニメ「連盟空軍航空魔法音楽隊 ルミナスウィッチーズ」PV第3弾 [AlxIeo7bvQE].mp4")

src_avc = initialise_clip(fileinfo.clip)
src_vp9 = initialise_clip(stg.src(rf"{src_dir}\TVアニメ「連盟空軍航空魔法音楽隊 ルミナスウィッチーズ」PV第3弾 [AlxIeo7bvQE].webm"))

vp9_protodeb = core.average.Mean([src_avc, src_vp9.dfttest.DFTTest(0, 1.6, 9, tbsize=5)] + [
    src_vp9.deblock.Deblock(quant) for quant in range(16, 32, 4)
])

src_vp9 = stg.other.bestframeselect([src_avc, src_vp9, vp9_protodeb], src_avc)
src_vp9 = core.std.Expr([src_vp9, src_avc], 'x y max')

freqmerge = stg.oyster.Core().FreqMerge(src_avc, src_vp9)
freqmerge = ccd(freqmerge, 5)

deblock = vsdpir(
    core.average.Mean([
        freqmerge, core.median.Median([
            contra_fdha(clip, clip.cas.CAS(0.175), 1.8) for clip in [
                src_avc, freqmerge, src_vp9
            ]
        ])
    ]), 75, backend=Backend.TRT
)

deblock = lvf.sraa(deblock, 1.65)

deband1 = f3kbilateral(deblock)

deband = core.average.Mean([
    deband1,
    dumb3kdb(deband1, 8, 18), dumb3kdb(deblock, 12, 24), dumb3kdb(deblock, 24, 34),
    placebo_deband(deband1, 8, grain=2), placebo_deband(deblock, 12, grain=3)
]).std.MaskedMerge(deblock, FDoGTCanny().edgemask(deblock))

deband = contrasharpening(deband, deband)

pref = iterate(get_y(deband), core.std.Maximum, 2).std.Convolution([1] * 9)

grain = Graigasm(
    thrs=[x << 8 for x in (40, 98, 160)],
    strengths=[(0.7, 0.15), (0.3, 0.1), (0.1, 0.06)],
    sizes=(0.8, 0.76, 0.65),
    sharps=(55, 45, 40),
    grainers=[
        AddGrain(seed=80085, constant=False),
        AddGrain(seed=69420, constant=True),
        AddGrain(seed=69420, constant=True)
    ]
).graining(deband, prefilter=pref)
grain = finalise_clip(grain)


if __name__ == "__main__":
    encoder = X265("x265_settings")
    encoder.resumable = True

    encoder.run_enc(grain, fileinfo, qpfile_clip=fileinfo.clip)
else:
    stg.output(src_avc)
    stg.output(src_vp9)
    stg.output(grain)
