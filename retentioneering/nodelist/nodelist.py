from __future__ import annotations

import numpy as np
import pandas as pd

from retentioneering.tooling.transition_graph.interface import (
    RenameRule,
    SerializedNodelist,
    ThresholdValue,
    ThresholdValueMap,
)

IS_HIDDEN_BY_USER_COL = "is_hidden_by_user"
PARENT_ID_COL = "parent_id"
IS_GROUP_COL = "is_group"
IS_AGGREGATED_COL = "is_aggregated"
NAME_COL = "name"
IS_OUT_OF_THRESHOLD_COL = "rete_is_out_of_threshold"


class Nodelist:
    nodelist_df: pd.DataFrame
    event_col: str
    time_col: str
    weight_cols: list[str]
    payload_df: pd.DataFrame | None

    def __init__(
        self, event_col: str, time_col: str, weight_cols: list[str] | None, nodelist_df: pd.DataFrame | None = None
    ) -> None:
        self.event_col = event_col
        self.time_col = time_col
        self.weight_cols = weight_cols if weight_cols is not None else []

        if nodelist_df is not None:
            self.validate_nodelist(nodelist_df)
            self.nodelist_df = nodelist_df
        else:
            self.nodelist_df = self.create_empty_nodelist_df()

    def get_threshold_min_max(self) -> ThresholdValueMap:
        result: ThresholdValueMap = {}

        for weight_col in self.weight_cols:
            min: float | int = self.nodelist_df[weight_col].min()
            max: float | int = self.nodelist_df[weight_col].max()
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

    def calculate_nodelist(
        self,
        data: pd.DataFrame,
        nodes_thresholds: ThresholdValueMap | None = None,
    ) -> pd.DataFrame:
        res: pd.DataFrame = data.groupby([self.event_col])[self.time_col].count().reset_index()

        for weight_col in self.weight_cols:
            by_col = data.groupby([self.event_col])[weight_col].nunique().reset_index()
            res = res.join(by_col[weight_col])
            res[weight_col] = res[weight_col].astype(float)

        res = res.sort_values(by=self.time_col, ascending=False)
        res = res.drop(columns=[self.time_col], axis=1)

        # just update weights
        res.set_index(self.event_col, inplace=True)
        self.nodelist_df.set_index(self.event_col, inplace=True)

        self.nodelist_df.update(res)
        res.reset_index(inplace=True)
        self.nodelist_df.reset_index(inplace=True)

        new_nodes = res[~res[self.event_col].isin(self.nodelist_df[self.event_col])]
        new_nodes = self.set_default_columns(new_nodes)

        if self.nodelist_df.empty:
            self.nodelist_df = new_nodes
        elif not new_nodes.empty:
            self.nodelist_df = pd.concat([self.nodelist_df, new_nodes])

        self.update_threshold(nodes_thresholds=nodes_thresholds)

        return self.nodelist_df

    def set_default_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df[IS_HIDDEN_BY_USER_COL] = False
        df[PARENT_ID_COL] = None
        df[IS_GROUP_COL] = False
        df[IS_AGGREGATED_COL] = False
        df[NAME_COL] = df[self.event_col]
        df[IS_OUT_OF_THRESHOLD_COL] = False

        df[IS_HIDDEN_BY_USER_COL] = df[IS_HIDDEN_BY_USER_COL].astype(bool)
        df[IS_GROUP_COL] = df[IS_GROUP_COL].astype(bool)
        df[IS_AGGREGATED_COL] = df[IS_AGGREGATED_COL].astype(bool)
        df[IS_OUT_OF_THRESHOLD_COL] = df[IS_OUT_OF_THRESHOLD_COL].astype(bool)

        return df

    def create_empty_nodelist_df(self) -> pd.DataFrame:
        columns = (
            [self.event_col]
            + self.weight_cols
            + [IS_HIDDEN_BY_USER_COL, PARENT_ID_COL, IS_GROUP_COL, IS_AGGREGATED_COL, NAME_COL, IS_OUT_OF_THRESHOLD_COL]
        )

        df = pd.DataFrame(columns=columns)

        df[IS_HIDDEN_BY_USER_COL] = df[IS_HIDDEN_BY_USER_COL].astype(bool)
        df[IS_GROUP_COL] = df[IS_GROUP_COL].astype(bool)
        df[IS_AGGREGATED_COL] = df[IS_AGGREGATED_COL].astype(bool)
        df[IS_OUT_OF_THRESHOLD_COL] = df[IS_OUT_OF_THRESHOLD_COL].astype(bool)

        return df

    def is_valid_nodelist_df(self, df: pd.DataFrame) -> bool:
        required_columns = [self.event_col] + [
            IS_HIDDEN_BY_USER_COL,
            PARENT_ID_COL,
            IS_GROUP_COL,
            IS_AGGREGATED_COL,
            NAME_COL,
        ]
        return all(column in df.columns for column in required_columns)

    def validate_nodelist(self, df: pd.DataFrame) -> None:
        if not self.is_valid_nodelist_df(df):
            raise ValueError("invalid nodelist_df")

    def groups_to_rename_rules(self) -> list[RenameRule]:
        rename_rules: list[RenameRule] = []
        df = self.nodelist_df
        groups: list[str] = df.loc[
            (df[IS_GROUP_COL] == True) & (df[IS_AGGREGATED_COL] == True), self.event_col
        ].tolist()

        for group in groups:
            childs = df.loc[df[PARENT_ID_COL] == group, self.event_col].tolist()
            if len(childs) > 0:
                rename_rules.append(dict(group_name=group, child_events=childs))

        return rename_rules

    def renamed_events_to_rename_rules(self) -> list[RenameRule]:
        rename_rules: list[RenameRule] = []

        df = self.get_ungrouped_nodes()
        renamed = df[df[self.event_col] != df[NAME_COL]]

        for _, row in renamed.iterrows():
            source_name = row[self.event_col]
            alias = row[NAME_COL]
            rename_rules.append(dict(group_name=alias, child_events=[source_name]))

        return rename_rules

    def update(self, df: pd.DataFrame) -> None:
        self.payload_df = df
        self.validate_nodelist(df)
        df[IS_OUT_OF_THRESHOLD_COL] = False
        cols = [self.event_col] + [
            IS_HIDDEN_BY_USER_COL,
            PARENT_ID_COL,
            IS_GROUP_COL,
            IS_AGGREGATED_COL,
            NAME_COL,
            IS_OUT_OF_THRESHOLD_COL,
        ]
        update_set = df[cols]

        groups_mask = update_set[IS_GROUP_COL] == True
        groups = update_set[groups_mask]

        self.nodelist_df.set_index(self.event_col, inplace=True)
        self.nodelist_df.update(groups.set_index(self.event_col))
        self.nodelist_df.reset_index(inplace=True)

        new_groups_mask = ~groups[self.event_col].isin(self.nodelist_df[self.event_col])
        new_groups = groups[new_groups_mask]

        for weight_col in self.weight_cols:
            new_groups[weight_col] = 0

        if not new_groups.empty:
            self.nodelist_df = pd.concat([self.nodelist_df, new_groups])

        events_nodes_mask = update_set[IS_GROUP_COL] == False
        events_nodes = update_set[events_nodes_mask]
        self.nodelist_df.set_index(self.event_col, inplace=True)
        self.nodelist_df.update(events_nodes.set_index(self.event_col))
        self.nodelist_df.reset_index(inplace=True)

    def update_threshold(self, nodes_thresholds: ThresholdValueMap | None = None) -> None:
        # filter nodes by threshold
        if nodes_thresholds:

            def is_out_of_threshold(row: pd.Series, column_name: str, threshold: ThresholdValue) -> bool:
                value = row[column_name]
                return not threshold["min"] <= value <= threshold["max"]

            self.nodelist_df[IS_OUT_OF_THRESHOLD_COL] = False

            for threshold_col, threshold in nodes_thresholds.items():
                if threshold_col in self.nodelist_df.columns:
                    self.nodelist_df[IS_OUT_OF_THRESHOLD_COL] = (
                        self.nodelist_df.apply(is_out_of_threshold, axis=1, args=(threshold_col, threshold))
                        | self.nodelist_df[IS_OUT_OF_THRESHOLD_COL]
                    )

    def get_ungrouped_nodes(self) -> pd.DataFrame:
        df = self.nodelist_df

        groups_roots_mask = (df[IS_GROUP_COL] == True) & (df[IS_AGGREGATED_COL] == True)
        not_grouped_mask = (df[IS_GROUP_COL] == False) & (df[PARENT_ID_COL].isna())

        parents = df.set_index(self.event_col)

        def parent_condition(
            row: pd.Series,
        ) -> bool:
            if pd.notna(row[PARENT_ID_COL]):
                parent = parents.loc[row[PARENT_ID_COL]]
                return parent[IS_GROUP_COL] == True and parent[IS_AGGREGATED_COL] == False
            return False

        grouped_but_not_aggregated_mask = (
            (df[IS_GROUP_COL] == False) & (df[PARENT_ID_COL].notna()) & df.apply(parent_condition, axis=1)
        )

        mask = groups_roots_mask | not_grouped_mask | grouped_but_not_aggregated_mask
        return df[mask]

    def get_out_of_threshold_nodes(self, only_ungrouped: bool = False) -> list[str]:
        df = self.get_ungrouped_nodes() if only_ungrouped else self.nodelist_df
        return df.loc[df[IS_OUT_OF_THRESHOLD_COL] == True, self.event_col].tolist()

    def get_hidden_by_user_nodes(self, groups: bool = False) -> list[str]:
        df = self.nodelist_df
        if groups == False:
            df = df[df[IS_GROUP_COL] == False]

        return df.loc[df[IS_HIDDEN_BY_USER_COL] == True, self.event_col].to_list()

    def get_min_max(self) -> ThresholdValueMap:
        result: ThresholdValueMap = {}

        if self.nodelist_df is None:  # type: ignore
            return result

        for weight_col in self.weight_cols:
            min = float(self.nodelist_df[weight_col].min())
            max = float(self.nodelist_df[weight_col].max())

            result[weight_col] = {"min": min, "max": max}

        return result

    def to_dict(self) -> SerializedNodelist:
        return self.nodelist_df.to_dict(orient="records")  # type: ignore

    def copy(self) -> Nodelist:
        weight_cols = self.weight_cols.copy()
        nodelist_df = self.nodelist_df.copy()

        return Nodelist(
            event_col=self.event_col,
            time_col=self.time_col,
            weight_cols=weight_cols,
            nodelist_df=nodelist_df,
        )
