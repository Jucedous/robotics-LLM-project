from __future__ import annotations
from typing import Any, Dict, Tuple
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button, TextBox




def setup_axes(
    ax: Axes,
    *,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    title: str,
    arena_face_color=(0.96, 0.97, 0.99),
    arena_edge_color=(0.75, 0.78, 0.85),
    grid_alpha: float = 0.35,
) -> None:
    ax.set_xscale("linear")
    ax.set_yscale("linear")

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_autoscale_on(False)
    ax.set_aspect("equal", adjustable="box")

    ax.set_title(title, color="black", fontsize=13, weight="bold")
    ax.set_facecolor(arena_face_color)
    ax.grid(True, alpha=grid_alpha)

    ax.add_patch(Rectangle(
        (xlim[0], ylim[0]),
        xlim[1] - xlim[0], ylim[1] - ylim[0],
        fill=False, lw=2.0, ec=arena_edge_color,
    ))

    ax.axhline(0.0, lw=0.8, ls="--", alpha=0.25)
    ax.axvline(0.0, lw=0.8, ls="--", alpha=0.25)

    fig = ax.figure
    def _keep_linear(ev):
        if ev.inaxes is ax and ev.key in ("l", "L"):
            ax.set_xscale("linear")
            ax.set_yscale("linear")
            fig.canvas.draw_idle()
    fig.canvas.mpl_connect("key_press_event", _keep_linear)




def setup_info_panel(fig: Figure, ax_info: Axes) -> Dict[str, Any]:
    ax_info.set_axis_off()
    score_box = Rectangle((0.05, 0.75), 0.90, 0.20, transform=ax_info.transAxes,
                          ec="black", lw=1.5, fc=(0.95, 0.95, 0.95))
    ax_info.add_patch(score_box)


    ax_info.text(0.50, 0.88, "Safety Score (0–5)",
                transform=ax_info.transAxes, ha="center", va="center",
                fontsize=12, color="black", weight="bold")


    txt_score_value = ax_info.text(0.50, 0.79, "—",
        transform=ax_info.transAxes,
        ha="center", va="center",
        fontsize=24, color="black", family="monospace")


    txt_info = ax_info.text(0.05, 0.70, "",
        transform=ax_info.transAxes,
        color="black", fontsize=10, family="monospace",
        ha="left", va="top")


    ax_close = fig.add_axes([0.72, 0.08, 0.20, 0.06])
    btn_close = Button(ax_close, "Close")


    return dict(score_box=score_box, txt_score_value=txt_score_value,
    txt_info=txt_info, btn_close=btn_close)




def setup_add_remove_controls(fig: Figure):
    # Returns (tb_name, tb_kind, btn_add, btn_del)
    gs = fig.add_gridspec(nrows=3, ncols=6, left=0.68, right=1.58, bottom=0.2, top=0.4, wspace=0.4, hspace=0.5)


    ax_name = fig.add_subplot(gs[0, 0])
    ax_kind = fig.add_subplot(gs[0, 1])


    tb_name = TextBox(ax_name, "name", initial="")
    tb_kind = TextBox(ax_kind, "kind", initial="")


    ax_add = fig.add_subplot(gs[1, 0:2])
    ax_del = fig.add_subplot(gs[2, 0:2])
    btn_add_obj = Button(ax_add, "Add object")
    btn_del_obj = Button(ax_del, "Remove by name")


    return tb_name, tb_kind, btn_add_obj, btn_del_obj