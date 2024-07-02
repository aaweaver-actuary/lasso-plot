"""Entrypoint for the FastAPI application."""

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import aiofiles
from dotenv import load_dotenv, find_dotenv

from lasso_plot.constants import FOLDER_WITH_HTML
from lasso_plot.lasso_path import LassoPath
from lasso_plot.path import LassoPathPlotter, LassoPath


load_dotenv(find_dotenv())

app = FastAPI()


@app.get("/lasso-path-data/{file}", response_class=HTMLResponse)
async def get_plot_from_server(file: str) -> HTMLResponse:
    """Read the HTML file and return it as a response."""
    async with aiofiles.open(f"{FOLDER_WITH_HTML}/{file}.html", mode="r") as f:
        html_content = await f.read()
    return HTMLResponse(
        filename=f"{FOLDER_WITH_HTML}/{file}.html", content=html_content
    )


@app.post("/lasso-path-data/{file}", response_class=HTMLResponse)
async def create_plot_with_params(file: str, params: dict) -> HTMLResponse:
    """Create a new HTML file with the specified parameters."""
    async with aiofiles.open(f"{FOLDER_WITH_HTML}/{file}.html", mode="w") as f:
        f.write(
            f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>LASSO Path</title>
            </head>
            <body>
                <h1>LASSO Path</h1>
                <p>Parameters: {params}</p>
            </body>
            </html>
            """
        )
    return HTMLResponse(
        filename=f"{FOLDER_WITH_HTML}/{file}.html",
        content=f"Created plot with parameters: {params}",
    )
