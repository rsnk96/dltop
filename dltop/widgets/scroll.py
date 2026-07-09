"""Vertical scroll region framed by small ``▴``/``▾`` "more content" edge hints."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

from textual.containers import Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.widgets import Static

if TYPE_CHECKING:
    from textual.app import ComposeResult
    from textual.widget import Widget


class _HintedInner(VerticalScroll):
    """Scrolling body that pokes its :class:`HintedScroll` parent to refresh hints.

    It fires whenever the reachable content changes so the parent can show or
    clear the ``▴``/``▾`` edge hints.
    """

    def on_mount(self) -> None:
        """Re-check the hints whenever the scroll offset moves."""
        self.watch(self, "scroll_y", self._notify, init=False)
        self.call_after_refresh(self._notify)

    def on_resize(self) -> None:
        """Re-check on resize, which can change whether the content overflows."""
        self._notify()

    def on_show(self) -> None:
        """Switching to this tab lays it out for the first time -- re-check."""
        self._notify()

    def _notify(self) -> None:
        parent = self.parent
        if isinstance(parent, HintedScroll):
            parent.update_hints(
                can_up=self.scroll_offset.y > 0,
                can_down=self.scroll_offset.y < self.max_scroll_y,
            )


class HintedScroll(Vertical):
    """A vertical scroll with a right-aligned ``▴``/``▾`` hint above and below.

    The hint rows are always exactly one line tall, so toggling their *text*
    can never change the inner scroll's height -- that would feed back into the
    overflow calculation and oscillate at the boundary. They simply show or
    clear the chevron as the scroll position allows.
    """

    DEFAULT_CSS = """
    HintedScroll { height: 1fr; }
    HintedScroll > .scroll-hint {
        height: 1;
        color: $accent;
        text-style: dim;
        text-align: right;
        padding: 0 2 0 0;
        background: $surface;
    }
    HintedScroll > .hinted-inner { height: 1fr; }
    """

    def __init__(self, *content: Widget) -> None:
        """Wrap the given ``content`` widgets in a hinted scroll region."""
        self._content = content
        super().__init__()

    def compose(self) -> ComposeResult:  # noqa: D102
        yield Static("", classes="scroll-hint scroll-hint-top")
        yield _HintedInner(*self._content, classes="tab-scroll hinted-inner")
        yield Static("", classes="scroll-hint scroll-hint-bottom")

    def update_hints(self, *, can_up: bool, can_down: bool) -> None:
        """Show ``▴``/``▾`` on whichever edges have more content past the fold."""
        with contextlib.suppress(NoMatches):
            self.query_one(".scroll-hint-top", Static).update("▴ more" if can_up else "")
            self.query_one(".scroll-hint-bottom", Static).update("▾ more" if can_down else "")
