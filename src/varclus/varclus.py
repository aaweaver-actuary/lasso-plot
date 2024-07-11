"""Define VarClus class for variable clustering."""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from joblib import Parallel, delayed, Memory
import logging
from varclus.src._bootstrap import varclus_bootstrap
from varclus.src._clustering import cluster_features
from varclus.src._force_diagram import plot_force_diagram
import plotly.graph_objects as go
from varclus.src.varclus_bootstrap_runner import VarclusBootstrapRunner
from varclus.src.varclus_clusterer import VarclusClusterer
from varclus.src.varclus_runner import VarclusRunner

__all__ = ["VarclusBootstrapRunner", "VarclusClusterer", "VarclusRunner"]

# Initialize the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create handlers
file_handler = logging.FileHandler("varclus.log")
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
        self, data: pd.DataFrame, n_clusters: int = 10, n_jobs: int = 4
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

    def run_bootstrap(self, n_iterations: int = 100) -> pd.DataFrame:
        """Run the VarClus bootstrap to estimate pairing probabilities."""
        logger.debug("Running VarClus bootstrap")
        prob_matrix = varclus_bootstrap(data=self.data, n_iterations=n_iterations)
        prob_matrix.to_parquet("pairing_probabilities.parquet")

    def read_probability_matrix(self) -> pd.DataFrame:
        """Read the pairing probabilities from disk."""
        try:
            return pd.read_parquet("pairing_probabilities.parquet")
        except FileNotFoundError:
            logger.error("Pairing probabilities not found. Running the bootstrap.")
            self.run_bootstrap()
            return pd.read_parquet("pairing_probabilities.parquet")

    def cluster_features(self) -> dict:
        """Cluster features based on their pairing probabilities."""
        probability_matrix = self.read_probability_matrix()
        logger.debug("Clustering features based on pairing probabilities")
        self.clusters = cluster_features(probability_matrix)
        return self.clusters

    def plot(self) -> go.Figure:
        """Plot a force-directed graph of the feature clusters."""
        if self.clusters is None:
            logger.error("Feature clusters have not been computed.")
            raise ValueError("Feature clusters have not been computed.")
        logger.debug("Plotting force-directed graph")
        return plot_force_diagram(self.clusters, self.data.columns)
