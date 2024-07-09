"""Layout component for the app."""

from __future__ import annotations
from dash import html

TAILWIND_CLASSES = "p-5 w-[70%]"


# Layout component wraps everything else
def Layout(children: list[html.Div]) -> html.Div:
    """Create the layout of the app."""
    return html.Div(children=children, className=TAILWIND_CLASSES, id="layout")
