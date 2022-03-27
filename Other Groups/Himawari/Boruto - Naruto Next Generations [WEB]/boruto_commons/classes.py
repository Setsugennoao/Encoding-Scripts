class Crop():
    frames = (0, 0)
    top = 0
    bottom = 0
    left = 0
    right = 0
    edgeFixing = True

    def __init__(self, frames, top=0, bottom=0, left=0, right=0, edgeFixing=True):
        self.frames = frames
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right
        self.edgeFixing = edgeFixing


class SimpleCrop(Crop):
    def __init__(self, frames):
        super().__init__(frames, 131, 131, 0, 0)
