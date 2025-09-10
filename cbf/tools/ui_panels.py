from __future__ import annotations
from typing import Any, Dict, Optional
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

    ax.set_title(title, color="black", fontsize=13, weight="bold", pad=8)
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

    ax_close = fig.add_axes([0.72, 0.03, 0.20, 0.06])
    btn_close = Button(ax_close, "Close")

    return dict(score_box=score_box, txt_score_value=txt_score_value,
                txt_info=txt_info, btn_close=btn_close)


def setup_add_remove_controls(fig: Figure, host_ax: Optional[Axes] = None):
    """
    Create 'Name' + 'Kind' TextBoxes and 'Add' + 'Remove' Buttons.

    If host_ax is provided, controls are packed neatly inside that axis'
    bounding box (figure coordinates). Otherwise we fall back to fixed
    figure-relative positions that won't overlap the info panel.

    Returns:
        (tb_name, tb_kind, btn_add_obj, btn_del_obj)
    """
    if host_ax is None:
        ax_name = fig.add_axes([0.72, 0.28, 0.24, 0.06])
        ax_kind = fig.add_axes([0.72, 0.20, 0.24, 0.06])
        ax_add  = fig.add_axes([0.72, 0.12, 0.115, 0.06])
        ax_del  = fig.add_axes([0.845, 0.12, 0.115, 0.06])
    else:

        host_ax.set_axis_off()

        bbox = host_ax.get_position()
        x0, y0, x1, y1 = bbox.x0, bbox.y0, bbox.x1, bbox.y1
        W, H = (x1 - x0), (y1 - y0)

        pad_lr = 0.06
        row_h  = 0.22
        gap    = 0.08

        top    = y0 + 0.88 * H
        y_name = top - row_h * H
        y_kind = y_name - gap * H - row_h * H
        y_btns = y_kind - gap * H - row_h * H

        ax_name = fig.add_axes([x0 + pad_lr * W, y_name, (1 - 2*pad_lr) * W, row_h * H])
        ax_kind = fig.add_axes([x0 + pad_lr * W, y_kind, (1 - 2*pad_lr) * W, row_h * H])
        half_w  = (1 - 2*pad_lr) * W * 0.5
        ax_add  = fig.add_axes([x0 + pad_lr * W,            y_btns, half_w - 0.01 * W, row_h * H])
        ax_del  = fig.add_axes([x0 + pad_lr * W + half_w + 0.01 * W, y_btns, half_w - 0.01 * W, row_h * H])

    tb_name = TextBox(ax_name, "Name", initial="")
    tb_kind = TextBox(ax_kind, "Kind", initial="")
    btn_add_obj = Button(ax_add, "Add")
    btn_del_obj = Button(ax_del, "Remove")

    for a in (ax_name, ax_kind, ax_add, ax_del):
        a.set_facecolor((0.98, 0.98, 0.99))
        for spine in a.spines.values():
            spine.set_alpha(0.15)

    return tb_name, tb_kind, btn_add_obj, btn_del_obj
