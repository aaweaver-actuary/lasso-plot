"""Define VarclusRunner class for variable clustering."""

from operator import is_
import numpy as np
import pandas as pd
from dataclasses import dataclass
from joblib import Parallel, delayed, Memory
from varclus.logger import logger
from varclus.src.clustering_algorithm import (
    ClusteringAlgorithm,
    SkAgglomerativeClusterer,
)
from varclus.src.dimensionality_reduction_algorithm import (
    DimensionalityReductionAlgorithm,
    SkPCA,
)

# Create a memory object to cache the results of the PCA step
memory = Memory(location="cache_dir", verbose=0)


# Define the function to run PCA on a chunk of data
@memory.cache
def reduce_dimensionality(
    data_chunk: pd.DataFrame, dim_reduction_algo: DimensionalityReductionAlgorithm
) -> np.ndarray:
    """
    Run a dimensionality reduction algorithm on a data chunk.

    Parameters
    ----------
    data_chunk : pd.DataFrame
        A DataFrame containing the data chunk to run the algorithm on.
    dim_reduction_algo : DimensionalityReductionAlgorithm
        An instance of a dimensionality reduction algorithm.

    Returns
    -------
    np.ndarray
        The transformed data chunk.
    """
    dim_reduction_algo.fit(data_chunk.values)
    return dim_reduction_algo.transform(data_chunk.values)


@dataclass
class VarclusResampler:
    """Create a new instance of VarclusRunner with a bootstrapped dataset."""

    data: pd.DataFrame
    n_jobs: int = 8

    def resample(
        self, fraction: float = 0.7, n_observations: int = 2000, **kwargs
    ) -> "VarclusRunner":
        """Draw a resampled dataset from the input data."""
        if not 0 < fraction <= 1:
            raise ValueError(f"Fraction must be between 0 and 1, but got {fraction}")

        if fraction < 1:
            data = self.data.sample(frac=fraction, replace=False, axis=0)

        resampled_data = data.sample(n=n_observations, replace=True, axis=0)
        return VarclusRunner(resampled_data, self.n_job, **kwargs)


class VarclusRunner:
    """Variable clustering using PCA and Agglomerative Clustering.

    Algorithm
    ---------
    1. Initial random assignment of variables to clusters.
    2. Form clusters using the first principal component of the features in that
       cluster.
    3. Reassign variables to the cluster whose first principal component is most
       correlated with the variable.
    4. Repeat steps 2-3 until convergence.
    5. For each cluster, consider whether it should be split into two:
        a. Calculate the first principal component and its eigenvalue for the variables
           in the cluster
        b. Check if the eigenvalue exceeds the `eigval_threshold` attribute or if
           if the proportion of variance explained by the first principal component
           is below the `variance_threshold` attribute
        c. If the eigenvalue exceeds the threshold or the proportion of variance
           explained is below the threshold, split the cluster into two:
            i.  Calculate the second principal component for the variables in the cluster
           ii.  Use the second principal component to split the cluster into two
                new clusters using the Agglomerative Clustering algorithm
          iii.  Recalculate the first principal component and eigenvalue for each new
                cluster to determine whether further splitting is necessary
           iv.  Repeat steps i-iii until convergence
        d. If the cluster should not be split, keep the cluster as is
        e. Repeat steps 5a-5d for each cluster
    6. Calculate the correlation matrix between the first principal components of the
       clusters and consider highly-correlated clusters as candidates for merging:
        a. If the correlation between two clusters exceeds the `correlation_threshold`
           attribute, merge the two clusters
        b. Calculate the first principal component and eigenvalue for the merged cluster
           to ensure that the merged cluster should remain merged or should be split
        c. Repeat steps 6a-6b until no further merging is possible
    7. Repeat steps 2-6 until convergence

    Attributes
    ----------
    data : pd.DataFrame
        The input data for clustering.
    clustering_algo : ClusteringAlgorithm
        Clustering algorithm instance.
    dim_reduction_algo : DimensionalityReductionAlgorithm
        Diminensionality reduction algorithm instance.
    n_jobs : int, optional
        The number of jobs for parallel processing (default is 8).
    clusters : np.ndarray
        The cluster labels for the variables.
    eigval_threshold : float, optional
        The eigenvalue threshold for splitting clusters (default is 1).
    variance_threshold : float, optional
        The variance threshold for splitting clusters (default is 0.6).
    correlation_threshold : float, optional
        The correlation threshold for merging clusters (default is 0.8).
    """

    def __init__(
        self,
        data: pd.DataFrame,
        clustering_algo: ClusteringAlgorithm,
        dim_reduction_algo: DimensionalityReductionAlgorithm,
        n_jobs: int = 8,
        clusters: np.ndarray = None,
        eigval_threshold: float = 1,
        variance_threshold: float = 0.6,
        correlation_threshold: float = 0.8,
    ) -> None:
        self.data = data
        self.clustering_algo = clustering_algo | SkAgglomerativeClusterer
        self.dim_reduction_algo = dim_reduction_algo | SkPCA
        self.n_jobs = n_jobs
        self.clusters = None

    def validate_data(self) -> None:
        """Validate the input data."""
        if self.data.isna().any():
            raise ValueError(
                "Input data contains null values. Please clean the data before running VarClus."
            )
        if not isinstance(self.data, pd.DataFrame):
            raise ValueError("Input data should be a pandas DataFrame.")

    def run(self, n_clusters: int) -> np.ndarray:
        """Run the VarClus algorithm to cluster the variables."""
        from varclus.src.varclus_helper_functions import (
            calculate_first_principal_component as get_first_pc,
            calculate_second_principal_component as get_second_pc,
            calculate_eigenvalues_of_covariance_matrix as get_eigenvalues,
        )

        # Validate the data
        self.validate_data()

        # Set the initial cluster assignments
        correlation_matrix = self.data.corr().to_numpy()
        self.clustering_algo.fit(correlation_matrix)
        initial_clusters = self.clustering_algo.predict(correlation_matrix)
        self.clusters = initial_clusters

        # Run through this loop, iteratively splitting, merging, and reassigning
        # clusters until no more changes are made
        is_converged = False

        while not is_converged:
            # For each cluster, calculate the first principal component and eigenvalue
            for _ in range(n_clusters - 1):
                refined_clusters = []
                for cluster in np.unique(self.clusters):
                    cluster_data = self.data.iloc[:, self.clusters == cluster]
                    if cluster_data.shape[1] > 1:
                        transformed_data = reduce_dimensionality(
                            cluster_data, self.dim_reduction_algo
                        )
                        eigenvalues = get_eigenvalues(transformed_data)
                        if eigenvalues[0] < self.eigval_threshold:
                            refined_clusters.append(cluster)
                        else:
                            new_clusters = self.clustering_algo.predict(
                                cluster_data.corr().values
                            )
                            refined_clusters.extend(
                                new_clusters + max(refined_clusters, default=0) + 1
                            )
                    else:
                        refined_clusters.append(cluster)
                self.clusters = np.array(refined_clusters)

        return self.clusters
