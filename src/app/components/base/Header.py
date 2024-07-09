"""Header components."""

from dash import html

# Alias all header types for the base_header function
HtmlHeaderType = html.H1 | html.H2 | html.H3 | html.H4 | html.H5 | html.H6


def base_header(level: int, text: str, *args) -> HtmlHeaderType:
    """Create an H1 header."""
    class_text = " ".join(args)
    return {
        1: html.H1(text, className=class_text),
        2: html.H2(text, className=class_text),
        3: html.H3(text, className=class_text),
        4: html.H4(text, className=class_text),
        5: html.H5(text, className=class_text),
        6: html.H6(text, className=class_text),
    }[level]


def h1(text: str, *args) -> html.H1:
    """Create an H1 header."""
    return base_header(1, text, "text-4xl", "font-bold", *args)


def h2(text: str, *args) -> html.H2:
    """Create an H2 header."""
    return base_header(2, text, "text-3xl", "font-bold", *args)


def h3(text: str, *args) -> html.H3:
    """Create an H3 header."""
    return base_header(3, text, "text-2xl", "font-bold", *args)


def h4(text: str, *args) -> html.H4:
    """Create an H4 header."""
    return base_header(4, text, "text-xl", "font-bold", *args)


def h5(text: str, *args) -> html.H5:
    """Create an H5 header."""
    return base_header(5, text, "text-lg", "font-bold", *args)


def h6(text: str, *args) -> html.H6:
    """Create an H6 header."""
    return base_header(6, text, "text-base", "font-bold", *args)
