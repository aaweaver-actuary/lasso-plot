from .embedded_plot import get_html_string
from fastapi import FastAPI, Response

app = FastAPI()


@app.get("/plots/top-25-glm-vs-full-model")
def plot_top_25_variable_glm_vs_full_model() -> Response:
    """Return an HTML webpage with the top 25 variables from the full model vs. the GLM model."""
    html_str = get_html_string()
    return Response(content=html_str, media_type="text/html")
