import stgfunc as stg
import vapoursynth as vs
from acsuite import eztrim


core = vs.core

path = r"."

src = stg.src(r".\Source\Johnny Test - 01 [jp dub] (Netflix 480p).mkv")

eztrim(
    src, (0, 1462),
    r".\Source\Johnny Test - 01 [jp dub] (Netflix 480p)_Audio01.JPN.wav",
    r".\Source\Johnny Test - 01 [jp dub] (Netflix 480p)_Audio01.JPN_cut_part1.wav",
)

eztrim(
    src, (1462, 9337),
    r".\Source\Johnny Test - 01 [jp dub] (Netflix 480p)_Audio01.JPN.wav",
    r".\Source\Johnny Test - 01 [jp dub] (Netflix 480p)_Audio01.JPN_cut_part3.wav",
)

eztrim(
    src, (9337, 15792),
    r".\Source\Johnny Test - 01 [jp dub] (Netflix 480p)_Audio01.JPN.wav",
    r".\Source\Johnny Test - 01 [jp dub] (Netflix 480p)_Audio01.JPN_cut_part6-7.wav",
)

eztrim(
    src, (29212, None),
    r".\Source\Johnny Test - 01 [jp dub] (Netflix 480p)_Audio01.JPN.wav",
    r".\Source\Johnny Test - 01 [jp dub] (Netflix 480p)_Audio01.JPN_cut_part8.wav",
)


# ffmpeg -i part1.wav -i part2.wav -i part3.wav -i part4.wav -i part5.wav -i part67.wav -i part8.wav -i part9.wav
# -filter_complex '[0:0][1:0][2:0][3:0]concat=n=8:v=0:a=1[out]'
# -map '[out]' output.wav
