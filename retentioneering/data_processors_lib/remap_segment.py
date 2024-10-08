from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from retentioneering.backend.tracker import collect_data_performance, time_performance
from retentioneering.data_processor import DataProcessor
from retentioneering.eventstream.segments import (
    SEGMENT_DELIMITER,
    SEGMENT_TYPE,
    _create_segment_event,
    _extract_segment_values,
    _get_segment_mask,
)
from retentioneering.eventstream.types import (
    AddSegmentType,
    EventstreamSchemaType,
    EventstreamType,
)
from retentioneering.params_model import ParamsModel
from retentioneering.utils.doc_substitution import docstrings
from retentioneering.utils.hash_object import hash_dataframe
from retentioneering.widget.widgets import ListOfString, ReteFunction

ADD_SEGMENT_TIMEDELTA = pd.Timedelta(1, "microseconds")


class RemapSegmentParams(ParamsModel):
    name: str
    mapping: dict


@docstrings.get_sections(base="RemapSegment")  # type: ignore
class RemapSegment(DataProcessor):
    """
    Remap segment values for synthetic eventstream events.

    Parameters
    ----------
    name : str
        A segment name to remap.
    mapping : str
        A pandas.Series containing mapping rules. The series index relates to old segment values,
         and the values relate to new segment values.

    Returns
    -------
    EventstreamType
        Eventstream with remapped segment.
    """

    params: RemapSegmentParams

    @time_performance(scope="remap_segment", event_name="init")
    def __init__(self, params: RemapSegmentParams) -> None:
        super().__init__(params=params)

    @time_performance(scope="remap_segment", event_name="apply")
    def apply(self, df: pd.DataFrame, schema: EventstreamSchemaType) -> pd.DataFrame:
        def mapper(value: Any) -> Any:
            return value if not mapping.get(value) else mapping.get(value)

        name = self.params.name
        mapping = self.params.mapping

        mask = _get_segment_mask(df, schema, name)
        values = _extract_segment_values(df.loc[mask, schema.event_name])

        df.loc[mask, schema.event_name] = _create_segment_event(values.map(mapper), name)

        # todo @dakhaytin - fill params
        collect_data_performance(scope="remap_segment", event_name="metadata", called_params={})
        return df
