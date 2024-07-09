# first line: 25
@memory.cache
def run_pca(data_chunk: pd.DataFrame) -> np.ndarray:
    """Run PCA on a chunk of data."""
    pca = PCA(n_components=1)
    return pca.fit_transform(data_chunk.T).T
