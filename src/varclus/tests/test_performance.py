import pytest
import pandas as pd
import numpy as np
from varclus.varclus import VarClus as Vc1
from varclus.varclus_v2 import VarClus as Vc2
import logging
import time

# Initialize the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create handlers
file_handler = logging.FileHandler("varclus_performance_optimization.log")
file_handler.setLevel(logging.INFO)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)

# Create formatters and add them to handlers
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)


@pytest.fixture
def data():
    rng = np.random.default_rng(42)
    return pd.DataFrame(rng.standard_normal((1000, 100)))


@pytest.mark.parametrize("VarClus", [Vc1, Vc2])
@pytest.mark.parametrize("n_clusters", [10, 20, 30])
def test_varclus(data, VarClus, n_clusters, n_jobs=4):
    logger.info(f"Running VarClus with {n_clusters} clusters and {n_jobs} jobs.")

    # Measure the time to create the VarClus object
    start_time = time.time()
    vc = VarClus(data, n_clusters=n_clusters, n_jobs=n_jobs)
    object_creation_time = time.time() - start_time
    logger.info(f"VarClus object created in {object_creation_time:.4f} seconds.")

    # Measure the overall time to run the clustering
    start_time = time.time()
    clusters = vc.run()
    total_runtime = time.time() - start_time
    logger.info(f"Clustering completed in {total_runtime:.4f} seconds.")

    # Additional detailed timing can be added inside the VarClus class if needed

    assert clusters is not None, f"Clusters are None for {VarClus}."
    assert (
        len(clusters) == data.shape[1]
    ), f"Number of clusters is incorrect for {VarClus}:\nExpected: {data.shape[1]}\nActual: {len(clusters)}"
