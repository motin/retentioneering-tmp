from typing import Callable, List, Optional, Tuple, Union, cast, get_args

import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from retentioneering.common.constants import (
    PATH_METRIC_TYPES,
    SCALERS,
    SEGMENT_METRICS_OVERVIEW_KINDS,
)
from retentioneering.common.formatters import fancy_round
from retentioneering.eventstream.segments import get_segment_path_map
from retentioneering.eventstream.types import EventstreamType


class SegmentOverview:
    def __init__(self, eventstream: EventstreamType):
        self.__eventstream = eventstream
        self.segment_name: str
        self.table_1: pd.DataFrame
        self.table_2: pd.DataFrame
        self.table_1_annot: pd.DataFrame
        self.table_2_annot: pd.DataFrame
        self.min_value: float = 0.0
        self.max_value: float = 1.0
        self.heatmap_axis: int

    def _get_feature_table(
        self,
        features: pd.DataFrame,
        path_names: pd.Series,
        aggfunc: Union[Callable, str] = "mean",
        scaler: SCALERS = "minmax",
        axis: int = 1,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if scaler == "minmax":
            scaler_instance = MinMaxScaler()
        elif scaler == "std":
            scaler_instance = StandardScaler()
        else:
            raise ValueError(f"Unknown scaler {scaler}. Available scalers: {get_args(SCALERS)}")

        scaler_hm = MinMaxScaler(feature_range=(self.min_value, self.max_value))

        scaled_features = pd.DataFrame(
            scaler_instance.fit_transform(features), index=features.index, columns=features.columns
        )

        # hm stands for heatmap
        hm = scaled_features.groupby(path_names).agg(aggfunc)

        if axis == 0:
            hm = hm.transpose()

        hm = pd.DataFrame(scaler_hm.fit_transform(hm), index=hm.index, columns=hm.columns)

        if axis == 1:
            hm = hm.transpose()

        hm_annot = features.groupby(path_names).agg(aggfunc).transpose().pipe(fancy_round)

        return hm, hm_annot

    def _get_metrics_table(
        self,
        path_names: pd.Series,
        metrics: Optional[List[Tuple[PATH_METRIC_TYPES, Union[str, Callable], str]]],
        scaled: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        segment_size_col = "segment size"
        if metrics is None:
            metrics = [("segment_size", "mean", segment_size_col)]
        else:
            has_segment_size = False
            for metric in metrics:
                if metric[0] == "segment_size":
                    segment_size_col = metric[2]
                    has_segment_size = True
            if not has_segment_size:
                metrics.append(("segment_size", "mean", segment_size_col))

        scaler_cm = MinMaxScaler(feature_range=(self.min_value, self.max_value))
        # cm stands for custom path_metrics
        cm = pd.DataFrame()

        for metric_item in metrics:
            path_metric, segment_agg_metric, metric_name = metric_item
            if isinstance(path_metric, str) and path_metric == "segment_size":
                segment_agg_values = path_names.value_counts(normalize=True)
            else:
                path_metric = (path_metric, "tmp_metric_name")
                path_values = self.__eventstream.path_metrics(path_metric)
                segment_agg_values = path_values.groupby(path_names).agg(segment_agg_metric)
            cm[metric_name] = segment_agg_values

        cm = cm.sort_values(by=segment_size_col, ascending=False)
        cm_annot = cm.transpose().pipe(fancy_round)

        if scaled:
            cm = pd.DataFrame(scaler_cm.fit_transform(cm), index=cm.index, columns=cm.columns)

        cm = cm.transpose()

        return cm, cm_annot

    def fit_clusters_overview(
        self,
        segment_name: str,
        features: pd.DataFrame,
        aggfunc: Union[Callable, str] = "mean",
        scaler: SCALERS = "minmax",
        metrics: Optional[List[Tuple[PATH_METRIC_TYPES, Union[str, Callable], str]]] = None,
        axis: int = 1,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        path_names = cast(pd.Series, get_segment_path_map(self.__eventstream, name=segment_name))
        path_names = path_names.reindex(features.index).rename(segment_name)

        hm, hm_annot = self._get_feature_table(features, path_names, aggfunc, scaler, axis)
        cm, cm_annot = self._get_metrics_table(path_names, metrics)

        # sorting the columns by segment size
        hm = hm[cm.columns]
        hm_annot = hm_annot[cm.columns]

        # heatmap_axis
        if axis == 0:
            cm = pd.DataFrame(np.zeros(cm.values.shape) + 0.5, index=cm.index, columns=cm.columns)

        self.table_1 = hm
        self.table_2 = cm
        self.table_1_annot = hm_annot
        self.table_2_annot = cm_annot
        self.segment_name = segment_name
        self.axis = axis

        return hm, cm, hm_annot, cm_annot

    def fit_segment_overview(
        self,
        segment_name: str,
        metrics: Optional[List[Tuple[PATH_METRIC_TYPES, Union[str, Callable], str]]],
    ) -> pd.DataFrame:
        path_names = get_segment_path_map(self.__eventstream, name=segment_name, index="path_id")
        path_names = path_names.rename(segment_name)  # type: ignore

        cm, cm_annot = self._get_metrics_table(path_names, metrics, scaled=False)  # type: ignore

        self.table_2 = cm
        self.table_2_annot = cm_annot

        return cm

    def plot_heatmap(self) -> plotly.graph_objects.Figure:
        cell_height, cell_width = 20, 35
        character_width = 7
        colorbar_width = 100
        title_height = bottom_height = 100
        xtick_height = max(self.table_1.index.map(len)) * character_width
        index_width = self.table_1.index.union(self.table_2.index).str.len().max() * character_width

        width = index_width + cell_width * len(self.table_1.columns) + colorbar_width
        height = (
            cell_height * (len(self.table_1.index) + len(self.table_2.index))
            + title_height
            + bottom_height
            + xtick_height
        )

        hm_height_proportion = len(self.table_1.index) / (len(self.table_1.index) + len(self.table_2.index))
        cm_height_proportion = len(self.table_2.index) / (len(self.table_1.index) + len(self.table_2.index))

        fig = make_subplots(
            rows=2,
            cols=1,
            specs=[[{}], [{}]],
            row_heights=[hm_height_proportion, cm_height_proportion],
            vertical_spacing=0.01,
            shared_xaxes=True,
        )

        features_table = go.Heatmap(
            z=self.table_1,
            y=self.table_1.index,
            x=self.table_1.columns.astype(str),
            coloraxis="coloraxis1",
            xgap=1,
            ygap=1,
            text=self.table_1_annot,
            texttemplate="%{text}",
            hovertemplate=f"{self.segment_name}: %{{x}}<br>feature: %{{y}}<extra></extra>",
        )

        fig.add_trace(features_table, row=1, col=1)

        metrics_table = go.Heatmap(
            z=self.table_2,
            y=self.table_2.index,
            x=self.table_2.columns.astype(str),
            coloraxis="coloraxis2",
            xgap=1,
            ygap=1,
            text=self.table_2_annot,
            texttemplate="%{text}",
            hovertemplate=f"{self.segment_name}: %{{x}}<br>metric: %{{y}}<extra></extra>",
        )
        fig.add_trace(metrics_table, row=2, col=1)

        colorscale1 = "RdYlBu_r"
        colorscale2 = colorscale1 if self.axis == 1 else "Gray"
        tickangle = 0 if self.table_1.index.map(len).max() <= 3 else -90

        fig.update_layout(
            width=width,
            height=height,
            title_text=f"{self.segment_name} segment overview",
            title_x=0.5,
            coloraxis1=dict(
                colorscale=colorscale1,
                colorbar=dict(tickvals=[self.min_value, self.max_value], ticktext=["Low", "High"]),
            ),
            coloraxis2=dict(colorscale=colorscale2, showscale=False),
        )
        fig.update_xaxes(tickangle=tickangle)

        return fig

    def plot_custom_metrics(self, axis: int, kind: SEGMENT_METRICS_OVERVIEW_KINDS) -> plotly.graph_objects.Figure:
        cm = self.table_2
        cm_annot = self.table_2_annot

        if kind == "bar":
            if axis == 0:
                cm = cm.transpose()
            fig = px.bar(cm, barmode="group")

        elif kind == "heatmap":
            scaler = MinMaxScaler()
            if axis == 0:
                hm = cm.transpose()
            else:
                hm = cm.copy()

            # a workaround for timedelta columns
            # after transposing a timedelta columns loses its original dtype
            for col in hm.columns:
                try:
                    hm[col] = pd.to_timedelta(hm[col]).dt.total_seconds()
                except ValueError:
                    pass

            hm = pd.DataFrame(scaler.fit_transform(hm), index=hm.index, columns=hm.columns)

            if axis == 0:
                hm = hm.transpose()

            data = go.Heatmap(
                z=hm,
                y=hm.index,
                x=hm.columns.astype(str),
                coloraxis="coloraxis1",
                xgap=1,
                ygap=1,
                text=cm_annot,
                texttemplate="%{text}",
            )
            fig = go.Figure(data=data)

            fig.update_layout(
                coloraxis1=dict(
                    colorscale="RdYlBu_r",
                    colorbar=dict(tickvals=[self.min_value, self.max_value], ticktext=["Low", "High"]),
                )
            )
        else:
            raise ValueError(f"Unknown kind {kind}. Available kinds: {get_args(SEGMENT_METRICS_OVERVIEW_KINDS)}")

        return fig

    @staticmethod
    def _get_hover_values(df: pd.DataFrame) -> pd.DataFrame:
        new_data = []
        for i, row_index in enumerate(df.index):
            new_row = []
            for j, col_name in enumerate(df.columns):
                cell_value = df.iloc[i, j]
                new_row.append(f"feature: {row_index}<br>cluster_id: {col_name}<br>value: {cell_value}")
            new_data.append(new_row)
        new_df = pd.DataFrame(new_data, index=df.index, columns=df.columns)
        return new_df

    @property
    def values(self) -> Optional[Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]]:
        if hasattr(self, "table_1_annot") and hasattr(self, "table_2_annot"):
            return self.table_1_annot, self.table_2_annot
        elif not hasattr(self, "table_1_annot") and hasattr(self, "table_2_annot"):
            return self.table_2_annot
        return None
