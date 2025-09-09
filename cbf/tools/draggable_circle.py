import numpy as np
from matplotlib.patches import Circle

class DraggableCircle:
    def __init__(self, artist: Circle, radius: float, kind: str, name: str):
        self.artist = artist
        self.radius = radius
        self.kind = kind
        self.name = name
        self.press_offset = None

    def contains(self, event) -> bool:
        if event.inaxes != self.artist.axes:
            return False
        contains, _ = self.artist.contains(event)
        return contains

    def on_press(self, event):
        if not self.contains(event):
            return
        x0, y0 = self.artist.center
        self.press_offset = (x0 - event.xdata, y0 - event.ydata)

    def on_motion(self, event):
        if self.press_offset is None or event.inaxes != self.artist.axes:
            return
        if event.xdata is None or event.ydata is None:
            return
        dx, dy = self.press_offset
        new_x = event.xdata + dx
        new_y = event.ydata + dy
        r = self.radius
        new_x = np.clip(new_x, -1.0 + r, 1.0 - r)
        new_y = np.clip(new_y, -1.0 + r, 1.0 - r)
        self.artist.center = (new_x, new_y)

    def on_release(self, event):
        self.press_offset = None