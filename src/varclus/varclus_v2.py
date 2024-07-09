"""Define VarClus class for variable clustering."""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from joblib import Parallel, delayed, Memory
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filemode="a",
    filename="varclus.log",
)

# Create a logger object
logger = logging.getLogger(__name__)

# Create a memory object to cache the results of the PCA step
memory = Memory(location="cache_dir", verbose=0)


# Define the PCA function outside the class
@memory.cache
def run_pca(data_chunk: pd.DataFrame) -> np.ndarray:
    """Run PCA on a chunk of data."""
    pca = PCA(n_components=1)
    return pca.fit_transform(data_chunk.T).T


class VarClus:
    """Variable clustering using PCA and Agglomerative Clustering."""

    def __init__(
        self, data: pd.DataFrame, n_clusters: int = 10, n_jobs: int = -1
    ) -> None:
        self.data = data
        self.n_clusters = n_clusters
        self.n_jobs = n_jobs
        self.clusters = None

    def validate_data(self) -> None:
        """Validate the input data."""
        if not isinstance(self.data, pd.DataFrame):
            logger.error("Input data must be a pandas DataFrame.")
            raise ValueError("Input data must be a pandas DataFrame.")
        if self.data.isna().any().any():
            logger.error("Input data contains missing values.")
            raise ValueError("Input data contains missing values.")
        if self.data.empty or len(self.data.columns) < 2:
            logger.error("Input data must contain at least two variables.")
            raise ValueError("Input data must contain at least two variables.")

    def run_clustering(self, pca_result: np.ndarray) -> np.ndarray:
        """Run Agglomerative Clustering on the PCA results."""
        clustering = AgglomerativeClustering(n_clusters=self.n_clusters)
        return clustering.fit_predict(pca_result.T)

    def initialize_clusters(self) -> np.ndarray:
        """Initialize the clusters by running PCA and clustering on the input data."""
        self.validate_data()
        data_chunks = np.array_split(self.data, self.n_jobs, axis=1)  # Split by columns
        pca_results = Parallel(n_jobs=self.n_jobs)(
            delayed(run_pca)(chunk) for chunk in data_chunks
        )
        pca_results = np.hstack(pca_results)

        self.clusters = self.run_clustering(pca_results)
        return self.clusters

    def run(self) -> np.ndarray:
        """Run the VarClus algorithm to cluster the variables."""
        logger.debug("Running VarClus")
        self.initialize_clusters()
        return self.clusters
