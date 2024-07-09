"""Reproduce the algorithm used by SAS PROC VARCLUS for variable clustering."""

from __future__ import annotations

import itertools
import logging
import math
from dataclasses import dataclass
from functools import wraps
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
from tqdm import tqdm

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filemode="a",
    filename="varclus.log",
)

logger = logging.getLogger(__name__)


def validate_data(func: Callable) -> Callable:
    """Decorator to validate input data."""  # noqa: D401

    @wraps(func)
    def wrapper(self, *args, **kwargs) -> Callable:  # noqa: ANN001
        if not isinstance(self.data, pd.DataFrame):
            logger.error("Input data must be a pandas DataFrame.")
            raise ValueError("Input data must be a pandas DataFrame.")
        if self.data.isna().any().any():
            logger.error("Input data contains missing values.")
            raise ValueError("Input data contains missing values.")
        if self.data.empty or len(self.data.columns) < 2:
            logger.error("Input data must contain at least two variables.")
            raise ValueError("Input data must contain at least two variables.")
        return func(self, *args, **kwargs)

    return wrapper


def validate_clusters_initialized(func: Callable) -> Callable:
    """Decorator to validate that clusters have been initialized."""  # noqa: D401

    @wraps(func)
    def wrapper(self, *args, **kwargs) -> Callable:  # noqa: ANN001
        if not self.are_clusters_initialized:
            logger.error(
                "Clusters have not been initialized. Call 'initialize_clusters()' first."
            )
        return func(self, *args, **kwargs)

    return wrapper


@dataclass
class MatrixCalculator:
    """Class to compute correlation and distance matrices.

    Parameters
    ----------
    data : pd.DataFrame
        The input data with variables as columns.
    """

    data: pd.DataFrame

    @validate_data
    def compute_corr_matrix(self) -> np.ndarray:
        """Compute the correlation matrix.

        Returns
        -------
        np.ndarray
            The correlation matrix.
        """
        logger.debug("Computing the correlation matrix.")
        return self.data.corr().to_numpy()

    @validate_data
    def compute_dist_matrix(self, corr_matrix: np.ndarray) -> np.ndarray:
        """Compute the distance matrix.

        Parameters
        ----------
        corr_matrix : np.ndarray
            The correlation matrix.

        Returns
        -------
        np.ndarray
            The distance matrix.
        """
        logger.debug("Computing the distance matrix.")
        return np.round(1 - np.abs(corr_matrix), 9)


@dataclass
class ClusterInitializer:
    """Class to initialize clusters using hierarchical clustering.

    Parameters
    ----------
    dist_matrix : np.ndarray
        The distance matrix.
    clustering_method : str, optional
        The clustering method to use. Default is "average".
    max_clusters : int, optional
        The maximum number of clusters. If None, determined by threshold.
    threshold : float, optional
        The threshold for cutting the dendrogram to form initial clusters. Default is 0.7.
    """

    dist_matrix: np.ndarray
    clustering_method: str = "average"
    max_clusters: int | None = None
    threshold: float = 0.7

    def perform_hierarchical_clustering(self) -> np.ndarray:
        """Perform hierarchical clustering on the distance matrix.

        Returns
        -------
        np.ndarray
            The linkage matrix.
        """
        logger.debug(
            "Entering ClusterInitializer.perform_hierarchical_clustering method."
        )
        return linkage(
            pd.Series(squareform(self.dist_matrix, checks=False)).fillna(0).to_numpy(),
            method=self.clustering_method,
        )

    def get_clusters(self, Z: np.ndarray) -> np.ndarray:
        """Get initial clusters.

        Parameters
        ----------
        Z : np.ndarray
            The linkage matrix.

        Returns
        -------
        np.ndarray
            Array of cluster labels.
        """
        logger.debug("Entering ClusterInitializer.get_clusters method.")
        if self.max_clusters:
            return fcluster(Z, self.max_clusters, criterion="maxclust")
        return fcluster(Z, self.threshold, criterion="distance")


@dataclass
class PCAHandler:
    """Class to handle PCA computations.

    Parameters
    ----------
    data : pd.DataFrame
        The input data with variables as columns.
    n_components : int, optional
        The number of components to keep in PCA. Default is 1.
    """

    data: pd.DataFrame
    n_components: int = 1

    def get_most_important_component(self, vars: List[str]) -> Tuple[np.ndarray, float]:
        """Get the most important component in a cluster.

        Parameters
        ----------
        vars : List[str]
            List of variables in the cluster.

        Returns
        -------
        Tuple[np.ndarray, float]
            The most important component and the explained variance ratio.
        """
        logger.debug("Entering PCAHandler.get_most_important_component method.")
        pca = PCA(n_components=self.n_components)
        data_subset = self.data[vars]
        pca.fit(data_subset)
        return pca.components_[0], pca.explained_variance_ratio_[0]

    def get_pca_variance(self, vars: List[str]) -> float:
        """Get the PCA variance in a cluster.

        Parameters
        ----------
        vars : List[str]
            List of variables in the cluster.

        Returns
        -------
        float
            The explained variance ratio of the most important component.
        """
        logger.debug("Entering PCAHandler.get_pca_variance method.")
        _, pca_variance = self.get_most_important_component(vars)
        return pca_variance


@dataclass
class ClusterMergerSplitter:
    """Class to handle merging and splitting of clusters.

    Parameters
    ----------
    data : pd.DataFrame
        The input data with variables as columns.
    clusters : np.ndarray
        Array of cluster labels.
    pca_handler : PCAHandler
        An instance of PCAHandler for PCA computations.
    """

    data: pd.DataFrame
    clusters: np.ndarray
    pca_handler: PCAHandler

    def get_variables(self, cluster: str | float) -> List[str]:
        """Get the variables in a cluster.

        Parameters
        ----------
        cluster : Any
            The cluster label.

        Returns
        -------
        List[str]
            List of variables in the cluster.
        """
        logger.debug("Entering ClusterMergerSplitter.get_variables method.")
        if cluster not in self.clusters:
            logger.error("Cluster label not found.")
            raise ValueError(
                f"Cluster label not found. Got: {cluster}, Expected one of: {self.clusters}"
            )
        return list(self.data.columns[self.clusters == cluster])

    def should_two_clusters_merge(
        self, cluster1: str | float, cluster2: str | float
    ) -> bool:
        """Return True if two clusters should merge based on PCA variance.

        Parameters
        ----------
        cluster1 : Any
            The first cluster label.
        cluster2 : Any
            The second cluster label.

        Returns
        -------
        bool
            True if the clusters should merge, otherwise False.
        """
        logger.debug("Entering ClusterMergerSplitter.should_two_clusters_merge method.")
        vars1 = self.get_variables(cluster1)
        vars2 = self.get_variables(cluster2)
        pca_variance1 = self.pca_handler.get_pca_variance(vars1)
        pca_variance2 = self.pca_handler.get_pca_variance(vars2)
        combined_vars = vars1 + vars2
        combined_variance = self.pca_handler.get_pca_variance(combined_vars)

        logger.debug(
            "Completed ClusterMergerSplitter.should_two_clusters_merge method."
        )
        return (combined_variance > pca_variance1) & (combined_variance > pca_variance2)

    def split_cluster(self, cluster: str | float) -> None:
        """Split a cluster based on PCA variance.

        Parameters
        ----------
        cluster : Any
            The cluster label.
        """
        logger.debug("Entering ClusterMergerSplitter.split_cluster method.")
        vars = self.get_variables(cluster)
        if len(vars) <= 1:
            return  # Cannot split a cluster with a single variable

        data_subset = self.data[vars]
        pca = PCA(n_components=2)
        pca.fit(data_subset)
        pca_components = pca.transform(data_subset)

        # Split based on the first principal component
        median_value = np.median(pca_components[:, 0])
        new_cluster = max(self.clusters) + 1

        logger.debug("Beggining loop in ClusterMergerSplitter.split_cluster method.")
        for var, value in zip(vars, pca_components[:, 0]):
            if value > median_value:
                self.clusters[self.data.columns == var] = new_cluster

        logger.debug("Completed ClusterMergerSplitter.split_cluster method.")


@dataclass
class VarClus:
    """Class to perform variable clustering using hierarchical clustering.

    Reproduces the algorithm used by SAS PROC VARCLUS.

    Parameters
    ----------
    data : pd.DataFrame
        The input data with variables as columns.
    max_clusters : int, optional
        The maximum number of clusters. If None, determined by threshold.
    n_components : int, optional
        The number of components to keep in PCA. Default is 1.
    threshold : float, optional
        The threshold for cutting the dendrogram to form initial clusters. Default is 0.7.
    clustering_method : str, optional
        The clustering method to use. Default is "average".
    """

    data: pd.DataFrame
    max_clusters: int | None = None
    n_components: int = 1
    threshold: float = 0.7
    clustering_method: str = "average"
    clusters: np.ndarray | None = None

    def __post_init__(self):
        """Initialize the class."""
        self.matrix_calculator = MatrixCalculator(self.data)
        self.pca_handler = PCAHandler(self.data, self.n_components)
        self.are_clusters_initialized = False
        self.final_clusters_ = None
        logger.debug("Initialized VarClus class.")

    def initialize_clusters(self) -> np.ndarray:
        """Initialize the clusters."""
        logger.debug("Entering VarClus.initialize_clusters method.")
        corr_matrix = self.matrix_calculator.compute_corr_matrix()
        dist_matrix = self.matrix_calculator.compute_dist_matrix(corr_matrix)
        logger.debug(f"Distance matrix:\n\n{dist_matrix}")
        cluster_initializer = ClusterInitializer(
            dist_matrix, self.clustering_method, self.max_clusters, self.threshold
        )
        Z = cluster_initializer.perform_hierarchical_clustering()
        clusters = cluster_initializer.get_clusters(Z)
        self.are_clusters_initialized = True
        self.clusters = clusters
        logger.debug("Completed VarClus.initialize_clusters method.")
        return clusters

    def get_clusters(self) -> pd.DataFrame:
        """Get the clusters."""
        logger.debug("Entering VarClus.get_clusters method.")
        if not self.are_clusters_initialized:
            logger.debug("Clusters have not been initialized. Initializing now.")
            self.initialize_clusters()

        logger.debug("Completed VarClus.get_clusters method.")
        return pd.DataFrame({"Variable": self.data.columns, "Cluster": self.clusters})

    def _one_iteration(self, cluster_merger_splitter: ClusterMergerSplitter) -> bool:
        """Perform one iteration of merging or splitting clusters."""
        logger.debug("Entering VarClus._one_iteration method.")
        initial_clusters = self.clusters.copy()

        logger.debug("Beggining loop in VarClus._one_iteration method.")
        n_relationships = math.comb(np.unique(self.clusters), 2)
        for i, j in tqdm(
            itertools.combinations(np.unique(self.clusters), 2),
            total=n_relationships,
            desc=f"Checking the {n_relationships} relationships between the clusters for any clusters that can be combined.\n",
        ):
            if cluster_merger_splitter.should_two_clusters_merge(i, j):
                print(f"Merging clusters {i} and {j}, with the new label being {i}")
                self.clusters[self.clusters == j] = i
                return True
        logger.debug("Completed loop in VarClus._one_iteration method.")

        logger.debug("Beggining loop in VarClus._one_iteration method.")
        for cluster in tqdm(
            np.unique(self.clusters),
            desc=f"Checking all the {np.unique(self.clusters)} clusters for any that can be split in two.\n",
        ):
            cluster_merger_splitter.split_cluster(cluster)
        logger.debug("Completed loop in VarClus._one_iteration method.")

        logger.debug("Completed VarClus._one_iteration method.")
        return not np.array_equal(initial_clusters, self.clusters)

    def run(self) -> pd.DataFrame:
        """Run the algorithm to get the final clusters."""
        logger.debug("Entering VarClus.run method.")
        if not self.are_clusters_initialized:
            logger.debug("Clusters have not been initialized. Initializing now.")
            self.initialize_clusters()

        cluster_merger_splitter = ClusterMergerSplitter(
            self.data, self.clusters, self.pca_handler
        )

        logger.debug("Beggining loop in VarClus.run method.")
        while self._one_iteration(cluster_merger_splitter):
            pass

        logger.debug("Completed loop in VarClus.run method. Assigning final clusters.")
        self.final_clusters_ = self.get_clusters()

        logger.debug("Completed VarClus.run method.")
        return self.final_clusters_

    @property
    def final_clusters(self) -> pd.DataFrame:
        """Get the final clusters."""
        logger.debug("Entering VarClus.final_clusters property.")
        if self.final_clusters_ is None:
            logger.debug("Final clusters have not been computed. Running now.")
            self.run()
            logger.debug("Final clusters have been computed.")

        logger.debug("Completed VarClus.final_clusters property.")
        return self.final_clusters_
