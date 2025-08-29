#!/usr/bin/env python3
# axes_with_numbers.py
# 3D space with long numeric axes. Zoom with scroll; rotate/pan disabled.

import numpy as np
import pyvista as pv

AXIS_LENGTH = 10000.0   # how far the axes extend
BG_COLOR = "white"

def lock_camera_except_zoom(plotter: pv.Plotter):
    """Disable rotate/pan but keep scroll zoom."""
    iren = plotter.iren
    def swallow(obj, evt):
        try: obj.SetAbortFlag(1)
        except Exception: pass
    for evt in (
        "LeftButtonPressEvent","LeftButtonReleaseEvent",
        "RightButtonPressEvent","RightButtonReleaseEvent",
        "MiddleButtonPressEvent","MiddleButtonReleaseEvent",
        "MouseMoveEvent",
    ):
        iren.add_observer(evt, swallow)

def main():
    L = AXIS_LENGTH
    p = pv.Plotter(window_size=(1280,800))
    p.set_background(BG_COLOR)

    # Show cube axes with numeric labels
    p.show_grid(
        bounds=(-L, L, -L, L, -L, L),  # x_min,x_max, y_min,y_max, z_min,z_max
        location="outer",
        color="black",
        font_size=12,
        n_xlabels=11, n_ylabels=11, n_zlabels=11,  # how many numeric ticks
        ticks="both",
        minor_ticks=True,
    )

    # Set a nice starting camera and clipping range
    p.camera_position = [(L*0.6, L*0.6, L*0.6), (0,0,0), (0,0,1)]
    p.camera.clipping_range = (0.01, L*10.0)

    # Lock camera so only zoom works
    lock_camera_except_zoom(p)

    p.add_text("Scroll to zoom. Axes are long and numbered.", color="black", font_size=12)

    p.show()

if __name__ == "__main__":
    main()
