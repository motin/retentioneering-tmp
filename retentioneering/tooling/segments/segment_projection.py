from typing import List, Optional

import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go

from retentioneering.eventstream.segments import get_all_segments
from retentioneering.eventstream.types import EventstreamType


class SegmentProjection:
    def __init__(self, eventstream: EventstreamType):
        self.__eventstream = eventstream
        self.segment_cols: List[str]
        self.proj: pd.DataFrame

    def fit(self, proj: pd.DataFrame, segments: Optional[List[str]] = None) -> None:
        path_segments = get_all_segments(self.__eventstream, resolve_collision="majority")

        if segments is not None:
            path_segments = path_segments[segments]

        proj = proj.join(path_segments, how="left").reset_index()

        self.proj = proj
        self.segment_cols = path_segments.columns.tolist()

    def plot(self) -> plotly.graph_objects.Figure:
        path_id_col = self.__eventstream.schema.user_id
        x_col = self.proj.columns[1]
        y_col = self.proj.columns[2]

        segment_values_nunique = self.proj[self.segment_cols].nunique()
        toggles_off = [False] * (segment_values_nunique.sum() + 1)

        default_toggles = toggles_off.copy()
        default_toggles[0] = True

        dropdowns = [
            {"label": "None", "method": "update", "args": [{"visible": default_toggles}, {"title": "2D-projection"}]}
        ]

        plot_data = px.scatter(self.proj, x=x_col, y=y_col, hover_data=[path_id_col]).data

        trace_index = 1
        for segment_name in self.segment_cols:
            data = px.scatter(self.proj, x=x_col, y=y_col, color=segment_name, hover_data=[path_id_col, segment_name])
            data.for_each_trace(lambda trace: trace.update(visible=False))
            plot_data += data.data

            segment_toggles = toggles_off.copy()
            for _ in range(segment_values_nunique[segment_name]):
                segment_toggles[trace_index] = True
                trace_index += 1

            dropdowns.append(
                {
                    "label": segment_name,
                    "method": "update",
                    "args": [{"visible": segment_toggles}, {"title": f"2D-projection<br>(by {segment_name})"}],
                }
            )

        fig = go.Figure(data=plot_data)

        fig.update_layout(
            width=900,
            height=800,
            title="2D-projection",
            title_x=0.5,
            updatemenus=[{"buttons": dropdowns, "x": 1.2, "y": 1.06}],
            annotations=[
                {"text": "Segment", "x": 1.11, "xref": "paper", "y": 1.1, "yref": "paper", "showarrow": False}
            ],
        )

        return fig

    @property
    def values(self) -> pd.DataFrame:
        return self.proj
