"""Define the dropdown component for the Dash app."""

from __future__ import annotations
from dash import dcc  # type: ignore
from typing import Dict
from uuid import uuid4


# we will define a default factory for the dictionary of options
def __default_options_factory() -> Dict[str, str]:
    """Return a default dictionary of options for the dropdown component."""
    return {"label": "OPTIONS DEFINED:", "value": "NONE DEFINED"}


def Dropdown(
    id: str | None = None,
    options: Dict[str, str] | None = None,
    initial_value: str | None = None,
    
) -> dcc.Dropdown:
    """Create a dropdown component for the Dash app."""
    if options is None:
        options = __default_options_factory()

    if initial_value is None:
        initial_value = options[next(iter(options.keys()))]

    return dcc.Dropdown(
        id=id if id is not None else f"dropdown-{uuid4()!s}",
        options=options,
        value=initial_value,
    )
