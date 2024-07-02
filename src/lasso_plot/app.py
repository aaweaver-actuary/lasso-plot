"""Entrypoint for the FastAPI application."""

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv, find_dotenv
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from lasso_plot.constants import FOLDER_WITH_HTML
from lasso_plot.lasso_path import LassoPath
from lasso_plot.path import LassoPathPlotter, LassoPath


def read_file(file: str) -> str:
    """Read the HTML file and return its content."""
    with Path(f"{FOLDER_WITH_HTML}/{file}.html", mode="r").open() as f:
        return f.read()


def write_file(file: str, content: str) -> None:
    """Write the content to the HTML file."""
    with Path(f"{FOLDER_WITH_HTML}/{file}.html", mode="w").open() as f:
        f.write(content)


async def async_read_file(file: str) -> str:
    """Read the HTML file asynchronously and return its content."""
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        return await loop.run_in_executor(pool, read_file, file)


async def async_write_file(file: str, content: str) -> None:
    """Write the content to the HTML file asynchronously."""
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        await loop.run_in_executor(pool, write_file, file, content)


load_dotenv(find_dotenv())

app = FastAPI()


@app.get("/lasso-path-data/{file}", response_class=HTMLResponse)
async def get_plot_from_server(file: str) -> HTMLResponse:
    """Read the HTML file and return it as a response."""
    html_content = await async_read_file(file)
    return HTMLResponse(
        filename=f"{FOLDER_WITH_HTML}/{file}.html", content=html_content
    )


@app.post("/lasso-path-data/{file}", response_class=HTMLResponse)
async def create_plot_with_params(file: str, params: dict) -> HTMLResponse:
    """Create a new HTML file with the specified parameters."""
    file_content = f"""
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
    await async_write_file(file, file_content)
    return HTMLResponse(
        filename=f"{FOLDER_WITH_HTML}/{file}.html",
        content=f"Created plot with parameters: {params}",
    )
