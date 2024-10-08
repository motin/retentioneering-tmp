from inspect import signature
from typing import Callable, Optional

from pandas import DataFrame, Series

from retentioneering.backend.tracker import (
    collect_data_performance,
    time_performance,
    track,
)
from retentioneering.data_processor import DataProcessor
from retentioneering.eventstream.schema import EventstreamSchema
from retentioneering.eventstream.segments import _filter_segment_events
from retentioneering.eventstream.types import (
    EventstreamSchemaType,
    SegmentFilterExpr,
    SplitExpr,
)
from retentioneering.params_model import ParamsModel
from retentioneering.utils.doc_substitution import docstrings
from retentioneering.utils.hash_object import hash_dataframe
from retentioneering.widget.widgets import ListOfString, ReteFunction


class FilterEventsParams(ParamsModel):
    """
    A class with parameters for :py:class:`.FilterEvents` class.

    """

    func: Optional[Callable[[DataFrame, Optional[EventstreamSchema]], Series]]
    segment: Optional[SegmentFilterExpr]

    _widgets = {"func": ReteFunction()}


@docstrings.get_sections(base="FilterEvents")  # type: ignore
class FilterEvents(DataProcessor):
    """
    Filter events from eventstream using a filtering function or a segment value.

    Parameters
    ----------
    func : Callable[[DataFrame, Optional[EventstreamSchema]], bool], optional
        Custom filtering function. It should accept input eventstream as a DataFrame
        and return a boolean mask of the same length indicating whether an event
        along with its row should be included in the output eventstream or not.

        You can also pass an additional EventstreamSchema parameter specifying the eventstream
        schema. For example, if ``event_timestamp`` is specified in the schema, you can access
        it as ``schema.event_timestamp`` regadless of the actual column name.

    segment : collection, optional
        A collection of two elements: a segment name and a segment value.

    Returns
    -------
    Eventstream
        Eventstream with filtered events.

    Examples
    --------
    Exclude events from a given list.

    .. code:: python

        users_to_exclude = [1, 2, 3]
        stream.filter_events(func=lambda df: ~df["user_id"].isin(users_to_exclude))

    Leave only events belonging to a specific segment.

    .. code:: python

        stream.filter_events(segment=['country', 'US'])

    Notes
    -----
    See :doc:`the data processor user guide</user_guides/dataprocessors>`
    and :doc:`the segment user guide</user_guides/segments_and_clusters>` for the details.
    """

    params: FilterEventsParams

    @time_performance(
        scope="filter_events",
        event_name="init",
    )
    def __init__(self, params: FilterEventsParams):
        if not params.func and not params.segment:
            raise KeyError("You have to pass at least one argument!")

        super().__init__(params=params)

    @time_performance(
        scope="filter_events",
        event_name="apply",
    )
    def apply(self, df: DataFrame, schema: EventstreamSchemaType) -> DataFrame:
        func: Optional[Callable[[DataFrame, Optional[EventstreamSchemaType]], Series]] = self.params.func  # type: ignore
        segment: Optional[SegmentFilterExpr] = self.params.segment  # type:ignore

        parent_shape = df.shape
        parent_hash = hash_dataframe(df)

        if segment:
            _filter_segment_events(df, schema, segment)

        if func:
            expected_args_count = len(signature(func).parameters)
            if expected_args_count == 1:
                mask = func(df)  # type: ignore
            else:
                mask = func(df, schema)

            df = df[mask]

        collect_data_performance(
            scope="filter_events",
            event_name="metadata",
            called_params=self.to_dict()["values"],
            performance_data={
                "parent": {
                    "shape": parent_shape,
                    "hash": parent_hash,
                },
                "child": {
                    "shape": df.shape,
                    "hash": hash_dataframe(df),
                },
            },
        )

        return df
