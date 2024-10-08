from __future__ import annotations

from typing import Callable, Literal, Union

import pandas as pd

METRIC_PREFIX_HAS = "has:"
METRIC_PREFIX_TIME_TO_EVENT = "time_to:"
CLUSTERING_METHODS = Literal["kmeans", "gmm", "hdbscan"]
# https://numpy.org/doc/stable/reference/arrays.datetime.html#datetime-units
DATETIME_UNITS = Literal["Y", "M", "W", "D", "h", "m", "s", "ms", "us", "μs", "ns", "ps", "fs", "as"]
DATETIME_UNITS_LIST = ["Y", "M", "W", "D", "h", "m", "s", "ms", "us", "μs", "ns", "ps", "fs", "as"]
FEATURE_TYPES = Literal["tfidf", "count", "frequency", "binary", "time", "time_fraction", "markov"]
NGRAM_SEP = "->"
PATH_METRIC_TYPES = Union[str, Callable, pd.NamedAgg]
PROJECTION_METHODS = Literal["tsne", "umap"]
SCALERS = Literal["minmax", "std"]
SEGMENT_METRICS_OVERVIEW_KINDS = Literal["heatmap", "bar"]
SEQUENCES_METRIC_TYPES = Literal["paths", "paths_share", "count", "count_share"]
SKLEARN_FEATURE_TYPES = Literal["count", "frequency", "tfidf", "binary"]
