import numpy as np
import pandas as pd
import polars as pl

ArrayLike = list | tuple | np.ndarray | pd.Series | pl.Series

__all__ = ["ArrayLike"]
