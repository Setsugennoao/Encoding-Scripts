import stgfunc as stg
import lvsfunc as lvf
import kagefunc as kgf
import vapoursynth as vs
from vsutil import plane, depth, join, get_y, iterate
from boruto_commons.utils import resize_spline, get_default_path, get_filenames
from boruto_commons.scenefilter import filterEnding16, filterPreview, filterTVTokyo
from boruto_commons.constants import FINAL_GRAIN_AMOUNT, ENCODING_x264_ARGS, TV_TOKYO_FRAMES, ED_16_FRAMES
from boruto_commons.filtering import knl_filter, descale_denoiseY_filter, debanding_filter, limit_yuv_filter, bil_downscale

ED_16_FRAMES = ED_16_FRAMES[1]

core = vs.core

filename = get_default_path(__file__)
filename_198 = get_default_path("198").replace('198\\', "Released\\198\\")

src = stg.src(filename, 16, matrix_prop=1)
src_198 = stg.src(filename_198, 16, matrix_prop=1)

src = lvf.rfs(src, core.std.StackVertical([
    src[2:].resize.Point(src_top=0, src_left=6.5).std.Crop(0, 0, 0, src.height - 2),
    src.std.Crop(0, 0, 0, 2)
]), (10251, 10252))

src = lvf.rfs(src, core.std.StackVertical([
    src[2:].resize.Point(src_top=1.575, src_left=6.5).std.Crop(0, 0, 0, src.height - 2),
    src.std.Crop(0, 0, 0, 2)
]), (10253, 10254))

src = lvf.rfs(src, core.std.StackVertical([
    src[10441].std.Crop(0, 0, 0, src.height - 2),
    src.std.Crop(0, 0, 0, 2)
]), (10442, 10445))

src = lvf.rfs(src, core.std.StackVertical([
    src[18251].std.Crop(0, 0, 0, src.height - 2),
    src.std.Crop(0, 0, 0, 2)
]), (18252, 18259))

ed_fuck_cred_mask = kgf.squaremask(src, 666, 696, 1140, 100)

src = lvf.rfs(src, src_198, [(32683, 32688), (32909, 32914)])
src = lvf.rfs(src, core.std.MaskedMerge(src_198, src, ed_fuck_cred_mask), (33299))
src = lvf.rfs(src, src.resize.Point(src_top=-2), (33300, 33303))

ed_fuck_cred_mask_2 = core.std.Expr([get_y(src), get_y(src_198)], 'x y - abs').std.Binarize(25 << 8)
ed_fuck_cred_mask_2 = iterate(iterate(ed_fuck_cred_mask_2, stg.Maximum, 10), core.std.Inflate, 500)

src = lvf.rfs(src, core.std.MaskedMerge(src_198, src, ed_fuck_cred_mask_2[32728]), (32711, 32713))

denoised_y, descale_mask = descale_denoiseY_filter(src)
bil_downscale = bil_downscale(src)

clip_y, clip_uv = knl_filter(denoised_y, 'Y'), knl_filter(src, 'UV')

denoised = join([
    resize_spline(clip_y),
    resize_spline(plane(clip_uv, 1)),
    resize_spline(plane(clip_uv, 2))
])

scenefiltered = filterTVTokyo(denoised, bil_downscale, TV_TOKYO_FRAMES)

scenefiltered = filterPreview(scenefiltered, bil_downscale, descale_mask, ED_16_FRAMES)

scenefiltered = filterEnding16(src, scenefiltered, ED_16_FRAMES)

deband = debanding_filter(scenefiltered)

grain = kgf.adaptive_grain(deband, FINAL_GRAIN_AMOUNT, True, 6.5)
grain = depth(grain, 10)
grain = limit_yuv_filter(grain)

if __name__ == '__main__':
  qpfilename, outfilename = get_filenames(filename)

  stg.encode.create_qpfile(src, qpfilename)
  stg.encode.encode(grain, outfilename, False, [], **ENCODING_x264_ARGS(qpfilename))
elif __name__ == '__vapoursynth__':
  grain.set_output()
else:
  stg.output(bil_downscale)
  stg.output(grain)
  stg.output(denoised)
  stg.output(scenefiltered)
  stg.output(deband)
