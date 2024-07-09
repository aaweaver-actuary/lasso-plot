"""Dropdown component."""

from __future__ import annotations
from dash import dcc, html


def Dropdown(
    id: str,
    features: list[str],
    placeholder: str | None = None,
    value: str | None = None,
) -> html.Div:
    """Create a dropdown component."""
    placeholder = features[0] if placeholder is None else placeholder
    value = placeholder if value is None else value
    return html.Div(
        [
            dcc.Dropdown(
                id=id,
                options=[{"label": i, "value": i} for i in features],
                placeholder=placeholder,
                value=value,
                className="""
            rounded-md border border-gray-300 shadow-sm
            focus:border-indigo-300 focus:ring focus:ring-indigo-200 focus:ring-opacity-50
            w-full

        """,
            )
        ],
        className="w-[80%] sm:w-[50%] md:w-[30%] lg:w-[20%] xl:w-[15%]",
    )
