from typing import Any, Literal, Optional, Tuple, cast

import pandas as pd
import umap
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.manifold import TSNE

from retentioneering.common.constants import (
    FEATURE_TYPES,
    NGRAM_SEP,
    PROJECTION_METHODS,
    SKLEARN_FEATURE_TYPES,
)
from retentioneering.eventstream.types import EventstreamType


class Vectorizer:
    def __init__(self, eventstream: EventstreamType):
        self.__eventstream = eventstream
        self.event_col = eventstream.schema.event_name
        self.time_col = eventstream.schema.event_timestamp
        self.event_index_col = eventstream.schema.event_index

    def _markov_vectorization(self, df: pd.DataFrame, path_id_col: str) -> pd.DataFrame:
        next_event_col = "next_" + self.event_col
        next_time_col = "next_" + self.time_col
        df = df.sort_values([path_id_col, self.event_index_col])
        df[[next_event_col, next_time_col]] = df.groupby(path_id_col)[[self.event_col, self.time_col]].shift(-1)
        embedding = (
            df.groupby([path_id_col, self.event_col, next_event_col])[self.event_index_col]
            .count()
            .reset_index()
            .rename(columns={self.event_index_col: "count"})
            .assign(bigram=lambda _df: _df[self.event_col] + NGRAM_SEP + _df[next_event_col])
            .assign(left_event_count=lambda _df: _df.groupby([path_id_col, self.event_col])["count"].transform("sum"))
            .assign(bigram_weight=lambda _df: _df["count"] / _df["left_event_count"])
            .pivot(index=path_id_col, columns="bigram", values="bigram_weight")
            .fillna(0)
        )
        embedding.index.rename(path_id_col, inplace=True)
        del df[next_event_col]
        del df[next_time_col]
        return embedding

    def _sklearn_vectorization(
        self,
        df: pd.DataFrame,
        feature_type: SKLEARN_FEATURE_TYPES,
        ngram_range: Tuple[int, int],
        path_id_col: str,
    ) -> pd.DataFrame:
        event_col = self.__eventstream.schema.event_name
        space_surrogate = "\t\t\t"

        def tokenizer(path: str) -> list:
            return path.split(NGRAM_SEP)

        paths = (
            # this trick is needed because get_feature_names_out can't use any other separator except space.
            df.assign(event=lambda _df: _df[event_col].str.replace(" ", space_surrogate))
            .groupby(path_id_col)[event_col]
            .apply(lambda x: NGRAM_SEP.join(x))
        )

        if feature_type == "tfidf":
            vectorizer = TfidfVectorizer(
                ngram_range=ngram_range, tokenizer=tokenizer, token_pattern=None, lowercase=False
            )
        elif feature_type in ["count", "frequency"]:
            vectorizer = CountVectorizer(
                ngram_range=ngram_range, tokenizer=tokenizer, token_pattern=None, lowercase=False
            )
        elif feature_type == "binary":
            vectorizer = CountVectorizer(
                ngram_range=ngram_range, tokenizer=tokenizer, token_pattern=None, binary=True, lowercase=False
            )
        else:
            raise ValueError(f"Unknown feature_type {feature_type}")

        embedding = vectorizer.fit_transform(paths)
        columns = vectorizer.get_feature_names_out()
        columns = [col.replace(" ", NGRAM_SEP).replace(space_surrogate, " ") for col in columns]
        embedding = pd.DataFrame(embedding.toarray(), index=paths.index, columns=columns)

        if feature_type == "frequency":
            row_sums = embedding.sum(axis=1)
            embedding = embedding.div(row_sums, axis=0).fillna(0)

        return embedding

    def extract_features(
        self,
        feature_type: FEATURE_TYPES,
        ngram_range: Tuple[int, int] = (1, 1),
        path_id_col: Optional[str] = None,
        col_suffix: Optional[str] = None,
    ) -> pd.DataFrame:
        if path_id_col is None:
            path_id_col = self.__eventstream.schema.user_id

        df = self.__eventstream.to_dataframe()

        if feature_type in ["count", "frequency", "tfidf", "binary"]:
            feature_type = cast(SKLEARN_FEATURE_TYPES, feature_type)
            embedding = self._sklearn_vectorization(df, feature_type, ngram_range, path_id_col)

        elif feature_type == "markov":
            embedding = self._markov_vectorization(df, path_id_col)

        elif feature_type in ["time", "time_fraction"]:
            df = df.sort_values(by=[path_id_col, self.time_col]).assign(
                time_to_next_event=lambda _df: -1
                * _df.groupby(path_id_col)[self.time_col].diff(periods=-1).dt.total_seconds()
            )
            # embedding for feature_type="time"
            embedding = df.groupby([path_id_col, self.event_col])["time_to_next_event"].sum().unstack(fill_value=0)

            if feature_type == "time_fraction":
                embedding = embedding.div(embedding.sum(axis=1), axis=0)
        else:
            raise ValueError(f"Unknown feature_type {feature_type}")

        if col_suffix is not None:
            embedding.columns = [f"{col}{col_suffix}" for col in embedding.columns]  # type: ignore

        return embedding

    @staticmethod
    def projection(
        features: pd.DataFrame,
        method: Literal["tsne", "umap"] = "tsne",
        sample_size: Optional[int] = None,
        random_state: Optional[int] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        if sample_size is not None:
            features = features.sample(n=sample_size, random_state=random_state)
        if method == "tsne":
            tsne_params = [
                "angle",
                "early_exaggeration",
                "init",
                "learning_rate",
                "method",
                "metric",
                "min_grad_norm",
                "n_components",
                "n_iter",
                "n_iter_without_progress",
                "n_jobs",
                "perplexity",
                "verbose",
            ]
            kwargs = {k: v for k, v in kwargs.items() if k in tsne_params}
            embedding = TSNE(random_state=random_state, **kwargs).fit_transform(features.values)

        elif method == "umap":
            reducer = umap.UMAP()
            umap_args_filter = reducer.get_params()
            kwargs = {k: v for k, v in kwargs.items() if k in umap_args_filter}
            embedding = umap.UMAP(random_state=random_state, **kwargs).fit_transform(features.values)
        else:
            raise ValueError(f"Unknown method: {method}. Allowed methods are: {PROJECTION_METHODS}")

        columns = [f"{method}_x", f"{method}_y"]
        embedding = pd.DataFrame(embedding, index=features.index, columns=columns)
        return embedding
