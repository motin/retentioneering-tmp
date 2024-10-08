from __future__ import annotations

import uuid
from typing import Optional, Tuple

import pandas as pd

from retentioneering.backend.tracker import collect_data_performance, time_performance
from retentioneering.data_processor import DataProcessor
from retentioneering.eventstream.segments import (
    SEGMENT_TYPE,
    _create_segment_event,
    _get_segment_mask,
)
from retentioneering.eventstream.types import (
    AddSegmentType,
    EventstreamSchemaType,
    EventstreamType,
)
from retentioneering.params_model import ParamsModel
from retentioneering.utils.doc_substitution import docstrings


class AddSegmentParams(ParamsModel):
    segment: AddSegmentType
    name: Optional[str]


@docstrings.get_sections(base="AddSegment")  # type: ignore
class AddSegment(DataProcessor):
    """
    Add segment synthetic events to eventstream.

    Parameters
    ----------
    segment : str, Callable, or pandas.Series
        Segment to add to eventstream.

        - If str, it should be a column name in eventstream.
        - If Callable, it should be a function that takes eventstream DataFrame and returns a list-like object specifing segment values for each event.
        - If pandas.Series, it should has index as path ids and values as segment values.
    name : str, optional
        Name of the segment.

    Returns
    -------
    EventstreamType
        Eventstream with added segment.
    """

    params: AddSegmentParams

    @time_performance(scope="add_segment", event_name="init")
    def __init__(self, params: AddSegmentParams) -> None:
        super().__init__(params=params)

    @time_performance(scope="add_segment", event_name="apply")
    def apply(self, df: pd.DataFrame, schema: EventstreamSchemaType) -> pd.DataFrame:
        orig_columns = df.columns
        segment = self.params.segment

        rows, name = self.calc_segment(df, segment, schema)

        mask = _get_segment_mask(df, schema, name, False)
        df.drop(df[mask].index, inplace=True)  # clean previous run

        segment_events = self.generate_segment_events(rows, name, schema)

        result = pd.concat([df[orig_columns], segment_events[orig_columns]])

        # todo: uncomment when enable data processors to delete columns
        if name in result.columns:
            result.drop(name, axis=1, inplace=True)
            schema.custom_cols = [col for col in schema.custom_cols if col != name]

        # todo @dakhaytin - fill params
        collect_data_performance(scope="add_segment", event_name="metadata", called_params={})
        return result

    def calc_segment(
        self, df: pd.DataFrame, segment: AddSegmentType, schema: EventstreamSchemaType
    ) -> Tuple[pd.DataFrame, str]:
        name: str
        rows: pd.DataFrame
        check_columns_conflict = True
        orig_columns = df.columns

        if isinstance(segment, str):
            if segment not in df.columns:
                raise KeyError("There is no such column in eventstream!")

            name = segment
            rows = self.get_segment_rows(df, name, schema)
            check_columns_conflict = False  # due to creation from column

        elif callable(segment):
            if not self.params.name:
                raise KeyError("You have to pass name if you want to create segment from lambda!")

            name = self.params.name
            df[name] = segment(df)
            rows = self.get_segment_rows(df, name, schema)

        elif isinstance(segment, pd.Series):  # pyright: ignore reportUnnecessaryIsInstance
            if not segment.name and not self.params.name:
                raise KeyError("You have to pass either Series with name or name to create segment!")

            name = segment.name or self.params.name  # type: ignore

            rows = df.groupby("user_id").first().reset_index()
            rows[name] = rows["user_id"].map(segment)

        else:
            raise ValueError("Wrong init format!")

        if check_columns_conflict and name in orig_columns:
            raise KeyError("Label should not conflict with eventstream exising columns!")

        return rows, name

    def get_segment_rows(self, df: pd.DataFrame, name: str, schema: EventstreamSchemaType) -> pd.DataFrame:
        tmp = df[[schema.user_id, schema.event_name, schema.event_type, name]].copy()
        tmp["shift"] = tmp[tmp[schema.event_type] == "raw"].groupby("user_id")[name].shift(1)

        rows = df.loc[(tmp[name] != tmp["shift"]) & (tmp[schema.event_type] == "raw")].copy()

        return rows

    def generate_segment_events(self, rows: pd.DataFrame, name: str, schema: EventstreamSchemaType) -> pd.DataFrame:
        event_col = schema.event_name
        type_col = schema.event_type
        event_index = schema.event_index
        id_col = schema.event_id

        rows[type_col] = SEGMENT_TYPE
        rows[event_col] = _create_segment_event(rows.loc[:, name], name)
        rows[event_index] = rows[event_index]
        rows[id_col] = [uuid.uuid4() for x in range(len(rows))]

        return rows
