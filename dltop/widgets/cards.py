"""Bordered key/value info card."""

from __future__ import annotations

from textual.widgets import Static


class InfoCard(Static):
    """Bordered key/value card.

    Content is a Rich-markup string; rows are updated atomically via
    :meth:`update_rows` so paint flicker is avoided.
    """

    DEFAULT_CSS = """
    InfoCard {
        border: round $accent;
        padding: 0 1;
        height: auto;
        width: 1fr;
        margin: 0 1 0 0;
        content-align: left top;
    }
    InfoCard:last-of-type {
        /* the margin above spaces cards apart; drop it on the last one so the row's
        right inset matches its left instead of doubling up with the row's own padding */
        margin-right: 0;
    }
    """

    def __init__(self, title: str, card_id: str) -> None:
        """Build a card with the given border ``title`` and widget ``card_id``."""
        super().__init__(id=card_id)
        self.border_title = title

    def update_rows(self, rows: list[tuple[str, str]]) -> None:
        """Repaint the card body with (label, value) pairs."""
        max_k = max((len(k) for k, _ in rows), default=0)
        body = "\n".join(f"[bold cyan]{k:<{max_k}}[/]  {v}" for k, v in rows)
        self.update(body)
