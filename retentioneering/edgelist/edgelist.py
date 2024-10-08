from __future__ import annotations

import pandas as pd

from retentioneering.eventstream.types import EventstreamType
from retentioneering.tooling.transition_graph.interface import (
    ThresholdValue,
    ThresholdValueMap,
)
from retentioneering.tooling.typing.transition_graph import NormType

IS_OUT_OF_THRESHOLD_COL = "rete_is_out_of_threshold"


class Edgelist:
    edgelist_df: pd.DataFrame
    eventstream: EventstreamType

    _weight_col: str

    def __init__(
        self,
        eventstream: EventstreamType,
        weight_cols: list[str] | None = None,
        edgelist_df: pd.DataFrame | None = None,
    ) -> None:
        self.eventstream = eventstream
        if edgelist_df is not None:
            self.edgelist_df = edgelist_df

        if weight_cols:
            self.weight_cols = weight_cols

    @property
    def weight_col(self) -> str:
        return self._weight_col

    @weight_col.setter
    def weight_col(self, value: str) -> None:
        if not value:
            raise ValueError("Weight col cannot be empty")
        self._weight_col = value

    @property
    def group_col(self) -> str:
        if self.weight_col in (self.eventstream.schema.event_id, self.eventstream.schema.user_id):
            return self.eventstream.schema.user_id
        else:
            return self.weight_col

    @property
    def next_event_col(self) -> str:
        return f"next_{self.eventstream.schema.event_name}"

    def calculate_edgelist(
        self,
        weight_cols: list[str],
        norm_type: NormType | None = None,
        edges_threshold: ThresholdValueMap | None = None,
    ) -> pd.DataFrame:
        self.weight_cols = weight_cols

        if norm_type not in (None, "full", "node"):
            raise ValueError(f"unknown normalization type: {norm_type}")

        edge_from, edge_to = self.eventstream.schema.event_name, self.next_event_col
        df = self.eventstream.to_dataframe()
        calculated_edgelist: pd.DataFrame = pd.DataFrame()
        for weight_col in weight_cols:
            self.weight_col = weight_col
            edgelist = self._calculate_edgelist_for_selected_weight(
                df=df, norm_type=norm_type, edge_from=edge_from, edge_to=edge_to
            )
            if calculated_edgelist.empty:
                calculated_edgelist = edgelist
            else:
                calculated_edgelist = self._merge_edgelist(calculated_edgelist, edge_from, edge_to, edgelist)

        self.edgelist_df = calculated_edgelist
        self.edgelist_df[IS_OUT_OF_THRESHOLD_COL] = False
        self.update_threshold(edges_threshold=edges_threshold)

        return calculated_edgelist

    def _merge_edgelist(
        self, calculated_edgelist: pd.DataFrame, edge_from: str, edge_to: str, edgelist: pd.DataFrame
    ) -> pd.DataFrame:
        calculated_edgelist = pd.merge(
            calculated_edgelist,
            edgelist,
            how="outer",
            left_on=[edge_from, edge_to],
            right_on=[edge_from, edge_to],
            suffixes=["", "_del"],
        )
        calculated_edgelist = calculated_edgelist[
            [c for c in calculated_edgelist.columns if not c.endswith("_del")]  # type: ignore
        ]
        return calculated_edgelist

    def _calculate_edgelist_for_selected_weight(
        self, df: pd.DataFrame, norm_type: NormType, edge_from: str, edge_to: str
    ) -> pd.DataFrame:
        possible_transitions = (
            df.assign(**{edge_to: lambda _df: _df.groupby(self.eventstream.schema.user_id)[edge_from].shift(-1)})
            .dropna(subset=[edge_to])
            .groupby([edge_from, edge_to])
            .size()
            .index
        )
        bigrams = df.assign(**{edge_to: lambda _df: _df.groupby(self.group_col)[edge_from].shift(-1)}).dropna(
            subset=[edge_to]
        )
        abs_values = bigrams.groupby([edge_from, edge_to])[self.weight_col].nunique()
        if self.weight_col != self.eventstream.schema.event_id:
            abs_values = abs_values.reindex(possible_transitions)
        edgelist = abs_values
        # denumerator_full = total number of transitions/users/sessions
        if norm_type == "full":
            if self.weight_col != "event_id":
                denumerator_full = bigrams[self.weight_col].nunique()
            else:
                denumerator_full = len(bigrams)
            edgelist = abs_values / denumerator_full
        # denumerator_node = total number of transitions/users/sessions that started with edge_from event
        if norm_type == "node":
            denumerator_node = bigrams.groupby([edge_from])[self.weight_col].nunique()
            edgelist = abs_values / denumerator_node
        if self.weight_col not in [self.eventstream.schema.event_id, self.eventstream.schema.user_id]:
            edgelist = edgelist.fillna(0)

            if norm_type is None:
                edgelist = edgelist.astype(int)
        edgelist = edgelist.reset_index()
        return edgelist

    def update_threshold(self, edges_threshold: ThresholdValueMap | None = None) -> None:
        # filter edges by threshold
        if edges_threshold:

            def is_out_of_threshold(row: pd.Series, column_name: str, threshold: ThresholdValue) -> bool:
                value = row[column_name]
                return not threshold["min"] <= value <= threshold["max"]

            self.edgelist_df[IS_OUT_OF_THRESHOLD_COL] = False

            for threshold_col, threshold in edges_threshold.items():
                if threshold_col in self.edgelist_df.columns:
                    self.edgelist_df[IS_OUT_OF_THRESHOLD_COL] = (
                        self.edgelist_df.apply(is_out_of_threshold, axis=1, args=(threshold_col, threshold))
                        | self.edgelist_df[IS_OUT_OF_THRESHOLD_COL]
                    )

    def get_filtered_edges(self) -> pd.DataFrame:
        filtered_edges = self.edgelist_df[~self.edgelist_df[IS_OUT_OF_THRESHOLD_COL]]
        filtered_edges = filtered_edges.drop(columns=[IS_OUT_OF_THRESHOLD_COL])
        return filtered_edges

    def get_min_max(self) -> ThresholdValueMap:
        result: ThresholdValueMap = {}

        if self.weight_cols is None:  # type: ignore
            return result

        if self.edgelist_df is None:  # type: ignore
            return result

        for weight_col in self.weight_cols:
            min = float(self.edgelist_df[weight_col].min())
            max = float(self.edgelist_df[weight_col].max())

            result[weight_col] = {"min": min, "max": max}

        return result

    def get_threshold_min_max(self) -> ThresholdValueMap:
        result: ThresholdValueMap = {}

        for weight_col in self.weight_cols:
            min: float | int = self.edgelist_df[weight_col].min()
            max: float | int = self.edgelist_df[weight_col].max()
            result[weight_col] = {
                "min": min,
                "max": max,
            }

        return result

    def fit_threshold(self, threshold: ThresholdValueMap, prev_min_max: ThresholdValueMap) -> ThresholdValueMap:
        new_threshold = threshold.copy()
        threshold_min_max = self.get_threshold_min_max()

        for weight_col, value in new_threshold.items():
            if weight_col not in prev_min_max:
                continue
            if weight_col not in threshold_min_max:
                continue

            if value["max"] == prev_min_max[weight_col]["max"]:
                new_threshold[weight_col]["max"] = threshold_min_max[weight_col]["max"]

        return new_threshold

    def copy(self) -> Edgelist:
        el = Edgelist(eventstream=self.eventstream, edgelist_df=self.edgelist_df.copy())

        if self.weight_cols is not None:  # type: ignore
            el.weight_cols = self.weight_cols.copy()

        if self.weight_col:
            el.weight_col = self.weight_col

        return el
