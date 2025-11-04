from __future__ import annotations
from typing import Callable, Optional
import getpass
import matplotlib.pyplot as plt

from matplotlib.widgets import TextBox, Button
from .feedback_parser import parse_feedback
from .preferences import PreferenceStore

def install_feedback_ui(
    fig: plt.Figure,
    on_recompute: Callable[[], None],
    user_id: Optional[str] = None,
    db_path: Optional[str] = None,
) -> None:

    user_id = user_id or getpass.getuser()

    ax_box = fig.add_axes([0.02, 0.02, 0.62, 0.05])
    tb = TextBox(ax_box, 'Feedback:', initial="")
    ax_btn = fig.add_axes([0.65, 0.02, 0.18, 0.05])
    btn = Button(ax_btn, 'Submit & Re-run')

    store = PreferenceStore(path=db_path) if db_path else PreferenceStore()

    def _on_submit(event=None):
        text = tb.text.strip()
        if not text:
            print("[feedback] empty input ignored")
            return
        try:
            rule = parse_feedback(text, user_id=user_id)
            rid = store.add_rule(rule)
            print(f"[feedback] saved rule {rid}: {rule}")
            tb.set_val("") 
            on_recompute()
        except Exception as e:
            print(f"[feedback] parse failed: {e}")

    btn.on_clicked(_on_submit)
    tb.on_submit(lambda _: _on_submit())

    fig.canvas.draw_idle()
