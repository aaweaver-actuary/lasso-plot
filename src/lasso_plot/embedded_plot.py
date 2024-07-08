"""Return an HTML document as a string with an iframe embedding an html file."""

IFRAME_CSS = """
    width: 100%;
    height: 100%;
    border: none;
"""

MAIN_CSS1 = "height: 100%;"
MAIN_CSS2 = """
    overflow: hidden;
    margin: 0;
    """


def get_html_string(
    iframe_src: str = "/user/aweaver/files/EGModeling/hit_ratio/bop_model/working_sessions/top_25_scatterplot__full_vs_GLM.html",
) -> str:
    """Return an HTML document as a string with an iframe embedding an html file."""
    return f"""
    <!DOCTYPE HTML>
    <html styles=>

    <head>
    <meta charset="utf-8">
    <title>top_25_scatterplot__full_vs_GLM.html</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    </head>

    <body>
    <style type="text/css">
        html, body, #container {
        {MAIN_CSS1}
        }
        body, #container {
        {MAIN_CSS2}
        }
        #iframe {
            {IFRAME_CSS}
        }
    </style>
    <div id="container">
        <iframe id="iframe" sandbox="allow-scripts" src={iframe_src}></iframe>
    </div>
    </body>

    </html>
    """
