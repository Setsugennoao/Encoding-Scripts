import vapoursynth as vs
from itertools import count
from stgfunc.misc import source

core = vs.core


EPS_SOURCES = [
    [
        source(r".\Source\01\Summer Time Rendering 01 - (Disney+).mkv", 16)[24:-24],
        source(r".\Source\02\Summer Time Rendering 02 - (Disney+).mkv", 16)[24:-24],
        source(r".\Source\03\Summer Time Rendering 03 - (Disney+).mkv", 16)[24:-24],
        source(r".\Source\04\Summer Time Rendering 04 - (Disney+).mkv", 16)[24:-24],
        source(r".\Source\05\Summer Time Rendering 05 - (Disney+).mkv", 16)[24:-24],
        source(r".\Source\06\Summer Time Rendering 06 - (Disney+).mkv", 16)[96:-24],
        source(r".\Source\07\Summer Time Rendering 07 - (Disney+).mkv", 16)[96:-24],
        source(r".\Source\08\Summer Time Rendering 08 - (Disney+).mkv", 16)[24:-24],
        source(r".\Source\09\Summer Time Rendering 09 - (Disney+).mkv", 16)[24:-24],
        source(r".\Source\10\Summer Time Rendering 10 - (Disney+).mkv", 16)[24:-24],
        source(r".\Source\11\Summer Time Rendering 11 - (Disney+).mkv", 16)[24:-24],
        source(r".\Source\12\Summer Time Rendering 12 - (Disney+).mkv", 16)[96:-24],
    ],
    [
        source(r".\Source\13\Summer Time Rendering 13 - (Disney+).mkv", 16)[24:-24],
    ]
]

EPS_OP_RANGES = [
    [
        (6882, 9037),
        (4507, 6665),
        (1200, 3356),
        (4507, 6665),
        (3045, 5202),
        (768, 2924),
        (1320, 3476),
        (3357, 5514),
        (2470, 4626),
        (0, 2155),
        (3117, 5274),
        (2182, 4338),
    ],
    [
        None,
    ]
]

EPS_ED_RANGES = [
    [
        (33685, 35841),
        (31529, 33686),
        None,
        (31576, 33685),
        (31527, 33684),
        (31529, 33686),
        (31528, 33685),
        (31528, 33685),
        (31527, 33684),
        (31528, 33685),
        (31528, 33685),
        (31528, 33685),
    ],
    [
        None
    ]
]


def merge_episodes(curr_ep_idx: int, cour: int) -> vs.VideoNode:
    episodes = [EPS_SOURCES[cour][curr_ep_idx]]
    EPISODE = episodes[0]
    OP_RANGES = EPS_OP_RANGES[cour][curr_ep_idx]
    ED_RANGES = EPS_ED_RANGES[cour][curr_ep_idx]

    assert len(EPS_SOURCES[cour]) == len(EPS_OP_RANGES[cour]) == len(EPS_ED_RANGES[cour])

    if not any({OP_RANGES, ED_RANGES}):
        return EPISODE

    for i, ep, rop, red in zip(count(), EPS_SOURCES[cour], EPS_OP_RANGES[cour], EPS_ED_RANGES[cour]):
        if i == curr_ep_idx or not any({rop, red}):
            continue

        episode = EPISODE

        if OP_RANGES and rop:
            len_op = OP_RANGES[1] + 1 - OP_RANGES[0]
            end_op = OP_RANGES[1] + 1

            insert_op = ep[rop[0]:rop[1] + 1]

            if insert_op.num_frames > len_op:
                insert_op = insert_op[:len_op]
            elif insert_op.num_frames < len_op:
                insert_op += episode[end_op - (len_op - insert_op.num_frames):end_op]

            if OP_RANGES[0]:
                episode = episode[:OP_RANGES[0]] + insert_op + episode[end_op:]
            else:
                episode = insert_op + episode[end_op:]

        if ED_RANGES and red:
            len_ed = ED_RANGES[1] + 1 - ED_RANGES[0]
            end_ed = ED_RANGES[1] + 1

            insert_ed = ep[red[0]:red[1] + 1]

            if insert_ed.num_frames > len_ed:
                insert_ed = insert_ed[:len_ed]
            elif insert_ed.num_frames < len_ed:
                insert_ed += episode[end_ed - (len_ed - insert_ed.num_frames):end_ed]
            episode = episode[:ED_RANGES[0]] + insert_ed + episode[end_ed:]

        episodes.append(episode)

    if not len(episodes):
        return EPS_SOURCES[cour][curr_ep_idx]

    if not len(episodes) % 2:
        episodes.append(EPISODE)

    if len(episodes) == 1:
        return episodes[0]

    return core.median.Median(episodes)
