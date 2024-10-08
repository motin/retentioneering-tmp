from typing import Any, Optional, Tuple, get_args

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from retentioneering.common.constants import CLUSTERING_METHODS, SCALERS
from retentioneering.eventstream.types import EventstreamType


class Clusters:
    def __init__(self, eventstream: EventstreamType):
        self.__eventstream = eventstream
        self.elbow_rule_max_clusters = 20

    def fit(
        self,
        X: pd.DataFrame,
        method: CLUSTERING_METHODS,
        n_clusters: Optional[int],
        scaler: Optional[SCALERS],
        random_state: Optional[int],
        segment_name: str = "cluster_id",
        **kwargs: Any,
    ) -> EventstreamType:
        if scaler is not None:
            if scaler == "minmax":
                scaler_instance = MinMaxScaler()
            elif scaler == "std":
                scaler_instance = StandardScaler()
            else:
                raise ValueError(f"Unknown scaler {scaler}. Available scalers: {get_args(SCALERS)}")
            X = scaler_instance.fit_transform(X)

        clusters_array = np.array([])
        if method == "kmeans":
            if n_clusters is not None:
                clusters_array, _ = self._kmeans(features=X, n_clusters=n_clusters, random_state=random_state, **kwargs)
            else:
                inertias_list = []
                n_range = list(range(1, self.elbow_rule_max_clusters))
                for n in n_range:
                    clusters_array, inertia = self._kmeans(
                        features=X, n_clusters=n, random_state=random_state, **kwargs
                    )
                    inertias_list.append(inertia)
                inertias = pd.Series(inertias_list, index=n_range)
                inertias.plot()

        elif method == "gmm":
            if n_clusters is not None:
                clusters_array = self._gmm(features=X, n_clusters=n_clusters, random_state=random_state, **kwargs)
            else:
                ValueError(f"You must provide a n_clusters argument for {method} method.")
        elif method == "hdbscan":
            clusters_array = self._hdbscan(features=X, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")

        labels = pd.Series(clusters_array, index=X.index, name=X.index.name)

        # remapping cluster labels according to the clusters size.
        # hdbscan associates -1 with a noise cluster. We keep -1 label from remapping since it's a special class.
        label_counts = labels.value_counts().drop(-1, errors="ignore")
        labels_remap = dict(zip(label_counts.index, range(len(label_counts))))
        labels = labels.replace(labels_remap)

        set_clusters = lambda _df: _df.merge(labels.to_frame(segment_name).reset_index())[segment_name]
        new_stream = self.__eventstream.add_segment(segment=set_clusters, name=segment_name)

        return new_stream

    @staticmethod
    def _kmeans(
        features: pd.DataFrame, random_state: Optional[int], n_clusters: int, **kwargs: Any
    ) -> Tuple[np.ndarray, float]:
        default_params = KMeans().get_params()
        kwargs = {k: v for k, v in kwargs.items() if k in default_params}
        model = KMeans(n_clusters=n_clusters, random_state=random_state, **kwargs)
        clusters = model.fit_predict(features.values)
        return clusters, model.inertia_

    @staticmethod
    def _gmm(features: pd.DataFrame, random_state: Optional[int], n_clusters: int, **kwargs: Any) -> np.ndarray:
        default_params = GaussianMixture().get_params()
        kwargs = {k: v for k, v in kwargs.items() if k in default_params}
        model = GaussianMixture(n_components=n_clusters, random_state=random_state, **kwargs)
        clusters = model.fit_predict(features.values)
        return clusters

    @staticmethod
    def _hdbscan(features: pd.DataFrame, **kwargs: Any) -> np.ndarray:
        default_params = HDBSCAN().get_params()
        kwargs = {k: v for k, v in kwargs.items() if k in default_params}
        model = HDBSCAN(**kwargs)
        clusters = model.fit_predict(features.values)
        return clusters
