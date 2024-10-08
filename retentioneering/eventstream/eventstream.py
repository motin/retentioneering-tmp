# flake8: noqa
from __future__ import annotations

import uuid
import warnings
from copy import deepcopy
from dataclasses import asdict
from types import FunctionType, LambdaType
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    MutableMapping,
    Optional,
    Tuple,
    Union,
    get_args,
)

import numpy as np
import pandas as pd
from IPython.display import HTML, display

from retentioneering.backend import counter
from retentioneering.backend.tracker import collect_data_performance, time_performance
from retentioneering.common.constants import (
    CLUSTERING_METHODS,
    DATETIME_UNITS,
    FEATURE_TYPES,
    METRIC_PREFIX_HAS,
    METRIC_PREFIX_TIME_TO_EVENT,
    PATH_METRIC_TYPES,
    SCALERS,
    SEGMENT_METRICS_OVERVIEW_KINDS,
    SEQUENCES_METRIC_TYPES,
)
from retentioneering.eventstream.schema import EventstreamSchema, RawDataSchema
from retentioneering.eventstream.types import (
    EventstreamType,
    RawDataCustomColSchema,
    RawDataSchemaType,
    SplitExpr,
    UserGroupsNamesType,
)
from retentioneering.path_metrics.vectorization import Vectorizer
from retentioneering.preprocessing_graph import PreprocessingGraph
from retentioneering.tooling import (
    Clusters,
    Cohorts,
    EventTimestampHist,
    Funnel,
    SegmentDiff,
    SegmentOverview,
    SegmentProjection,
    Sequences,
    StatTests,
    StepMatrix,
    StepSankey,
    TimedeltaHist,
    TransitionGraph,
    TransitionMatrix,
    UserLifetimeHist,
)
from retentioneering.tooling._describe import _Describe
from retentioneering.tooling._describe_events import _DescribeEvents
from retentioneering.tooling.constants import BINS_ESTIMATORS
from retentioneering.tooling.stattests.constants import STATTEST_NAMES
from retentioneering.tooling.timedelta_hist.constants import (
    AGGREGATION_NAMES,
    EVENTSTREAM_GLOBAL_EVENTS,
)
from retentioneering.tooling.typing.transition_graph import NormType, Threshold
from retentioneering.utils import get_merged_col
from retentioneering.utils.hash_object import hash_dataframe

from ..nodelist import Nodelist
from .helpers import (
    AddNegativeEventsHelperMixin,
    AddPositiveEventsHelperMixin,
    AddSegmentHelperMixin,
    AddStartEndEventsHelperMixin,
    CollapseLoopsHelperMixin,
    DropPathsHelperMixin,
    DropSegmentHelperMixin,
    FilterEventsHelperMixin,
    GroupEventsBulkHelperMixin,
    GroupEventsHelperMixin,
    LabelCroppedPathsHelperMixin,
    LabelLostUsersHelperMixin,
    LabelNewUsersHelperMixin,
    MaterializeSegmentHelperMixin,
    PipeHelperMixin,
    RemapSegmentHelperMixin,
    RenameHelperMixin,
    RenameSegmentHelperMixin,
    SplitSessionsHelperMixin,
    TruncatePathsHelperMixin,
)
from .segments import SEGMENT_TYPE, get_segment_path_map

OrderList = List[Optional[str]]
OrderDict = Dict[str, int]
FeatureType = Literal["tfidf", "count", "frequency", "binary", "time", "time_fraction", "external"]
NgramRange = Tuple[int, int]
Method = Literal["kmeans", "gmm"]
MetricsType = Literal["paths", "paths_share", "count", "count_share"]

DEFAULT_INDEX_ORDER: OrderList = [
    "profile",
    "segment",
    "path_start",
    "new_user",
    "existing_user",
    "cropped_left",
    "session_start",
    "session_start_cropped",
    "group_alias",
    "raw",
    "raw_sleep",
    None,
    "synthetic",
    "synthetic_sleep",
    "positive_target",
    "negative_target",
    "session_end_cropped",
    "session_end",
    "session_sleep",
    "cropped_right",
    "absent_user",
    "lost_user",
    "path_end",
]


def dictify(obj: Union[list, dict]) -> OrderDict:
    if isinstance(obj, list):
        return {obj[i]: i for i in range(len(obj))}

    return deepcopy(obj)


# @TODO: проработать резервирование колонок


class Eventstream(
    CollapseLoopsHelperMixin,
    DropPathsHelperMixin,
    FilterEventsHelperMixin,
    GroupEventsHelperMixin,
    GroupEventsBulkHelperMixin,
    LabelLostUsersHelperMixin,
    AddNegativeEventsHelperMixin,
    LabelNewUsersHelperMixin,
    AddPositiveEventsHelperMixin,
    SplitSessionsHelperMixin,
    AddStartEndEventsHelperMixin,
    LabelCroppedPathsHelperMixin,
    TruncatePathsHelperMixin,
    RenameHelperMixin,
    PipeHelperMixin,
    AddSegmentHelperMixin,
    DropSegmentHelperMixin,
    EventstreamType,
    RenameSegmentHelperMixin,
    RemapSegmentHelperMixin,
    MaterializeSegmentHelperMixin,
):
    """
    Collection of tools for storing and processing clickstream data.

    Parameters
    ----------
    raw_data : pd.DataFrame or pd.Series
        Raw clickstream data.
    raw_data_schema : dict or RawDataSchema, optional
        Represents mapping rules connecting important eventstream columns with the raw data columns.
        The keys are defined in :py:class:`.RawDataSchema`. The values are the corresponding column names
        in the raw data. ``custom_cols`` key stands for the defining additional columns that can be used in
        the eventstream. See the :ref:`Eventstream user guide <eventstream_raw_data_schema>` for the details.

    schema : dict or EventstreamSchema, optional
        Represents a schema of the created eventstream. The keys are defined in
        :py:class:`.EventstreamSchema`. The values are the names of the corresponding eventstream columns.
        See the :ref:`Eventstream user guide <eventstream_field_names>` for the details.

    custom_cols : list of str, optional
        The list of additional columns from the raw data to be included in the eventstream.
        If not defined, all the columns from the raw data are included.

    prepare : bool, default True
        - If ``True``, input data will be transformed in the following way:

            - ``event_timestamp`` column is converted to pandas datetime format.
            - ``event_type`` column is added and filled with ``raw`` value.
              If the column exists, it remains unchanged.

        - If ``False`` - ``raw_data`` will be remained as is.

    index_order : list of str, default DEFAULT_INDEX_ORDER
        Sorting order for ``event_type`` column.
    user_sample_size : int of float, optional
        Number (``int``) or share (``float``) of all users' trajectories that will be randomly chosen
        and left in final sample (all other trajectories will be removed) .
        See :numpy_random_choice:`numpy documentation<>`.
    user_sample_seed : int, optional
        A seed value that is used to generate user samples.
        See :numpy_random_seed:`numpy documentation<>`.
    events_order : list of str, optional
        Sorting order for ``event_name`` column, if there are events with equal timestamps inside each user trajectory.
        The order of raw events is fixed once while eventstream initialization.
    add_start_end_events : bool, default True
        If True, ``path_start`` and ``path_end`` synthetic events are added to each path explicitly.
        See also :py:class:`.AddStartEndEvents` documentation.
    convert_tz : 'local' or 'UTC', optional
        Timestamp column with timezones is not supported in the eventstream and should be explicitly converted.

        - If ``UTC``, the timestamp column will be converted to utc time, and the timezone part will be truncated.
        - If ``local``, the timezone will be truncated.

    Notes
    -----
    See :doc:`Eventstream user guide</user_guides/eventstream>` for the details.


    """

    schema: EventstreamSchema
    events_order: OrderDict
    index_order: OrderDict
    __hash: str = ""
    _preprocessing_graph: PreprocessingGraph | None = None

    __raw_data_schema: RawDataSchemaType
    __events: pd.DataFrame | pd.Series[Any]
    __funnel: Funnel
    __clusters: Clusters
    __segment_overview: SegmentOverview
    __cohorts: Cohorts
    __step_matrix: StepMatrix
    __sankey: StepSankey
    __stattests: StatTests
    __sequences: Sequences
    __transition_graph: TransitionGraph
    __transition_matrix: TransitionMatrix
    __timedelta_hist: TimedeltaHist
    __user_lifetime_hist: UserLifetimeHist
    __event_timestamp_hist: EventTimestampHist
    __eventstream_index: int

    @time_performance(
        scope="eventstream",
        event_name="init",
    )
    def __init__(
        self,
        raw_data: pd.DataFrame | pd.Series[Any],
        raw_data_schema: RawDataSchema
        | RawDataSchemaType
        | dict[str, str | list[RawDataCustomColSchema]]
        | None = None,
        schema: EventstreamSchema | dict[str, str | list[str]] | None = None,
        prepare: bool = True,
        index_order: Optional[Union[OrderList, OrderDict]] = None,
        user_sample_size: Optional[int | float] = None,
        user_sample_seed: Optional[int] = None,
        events_order: Optional[Union[OrderList, OrderDict]] = None,
        custom_cols: List[str] | None = None,
        add_start_end_events: bool = True,
        convert_tz: Optional[Literal["UTC", "local"]] = None,
        segment_cols: Optional[List[str]] = None,
    ) -> None:
        tracking_params = dict(
            raw_data=raw_data,
            prepare=prepare,
            index_order=index_order,
            user_sample_size=user_sample_size,
            user_sample_seed=user_sample_seed,
            events_order=events_order,
            custom_cols=custom_cols,
            add_start_end_events=add_start_end_events,
            convert_tz=convert_tz,
            segment_cols=segment_cols,
        )
        not_hash_values = ["raw_data_schema", "schema", "convert_tz", "index_order", "events_order", "segment_cols"]

        if not schema:
            schema = EventstreamSchema()
        elif isinstance(schema, dict):
            schema = EventstreamSchema(**schema)  # type: ignore

        self.schema = schema
        self.__eventstream_index: int = counter.get_eventstream_index()

        if not raw_data_schema:
            raw_data_schema = RawDataSchema()
            if self.schema.event_type in raw_data.columns:
                raw_data_schema.event_type = self.schema.event_type

        elif isinstance(raw_data_schema, dict):
            raw_data_schema = RawDataSchema(**raw_data_schema)  # type: ignore
        self.__raw_data_schema = raw_data_schema

        if custom_cols is None and not self.__raw_data_schema.custom_cols and not self.schema.custom_cols:
            custom_cols = self.__define_default_custom_cols(raw_data=raw_data)

        if custom_cols and prepare:
            self.__raw_data_schema.custom_cols = []
            self.schema.custom_cols = []
            for col_name in custom_cols:
                col: RawDataCustomColSchema = {"raw_data_col": col_name, "custom_col": col_name}
                self.__raw_data_schema.custom_cols.append(col)
                self.schema.custom_cols.append(col_name)

        self.convert_tz = convert_tz

        raw_data_schema_default_values = asdict(RawDataSchema())
        schema_default_values = asdict(EventstreamSchema())
        if isinstance(raw_data_schema, RawDataSchema):
            tracking_params["raw_data_schema"] = [
                key for key, value in raw_data_schema_default_values.items() if asdict(raw_data_schema)[key] != value
            ]
        tracking_params["schema"] = [
            key for key, value in schema_default_values.items() if asdict(self.schema)[key] != value
        ]

        self.__track_dataset(
            name="metadata",
            data=raw_data,
            params=tracking_params,
            schema=self.__raw_data_schema,
            not_hash_values=not_hash_values,
        )

        if user_sample_size is not None:
            raw_data = self.__sample_user_paths(raw_data, raw_data_schema, user_sample_size, user_sample_seed)
        if not index_order:
            self.index_order = dictify(DEFAULT_INDEX_ORDER)
        else:
            self.index_order = dictify(index_order)

        if events_order is not None:
            self.events_order = dictify(events_order)
        else:
            self.events_order = dict()

        self.__events = self.__prepare_events(raw_data) if prepare else raw_data
        self.__events = self.__required_cleanup(events=self.__events)
        self.__apply_default_dataprocessors(add_start_end_events=add_start_end_events, segment_cols=segment_cols)
        self.index_events()

        if prepare:
            self.__track_dataset(
                name="metadata",
                data=self.__events,
                params=tracking_params,
                schema=self.schema,
                not_hash_values=not_hash_values,
            )

        self._preprocessing_graph = None

    @property
    def _eventstream_index(self) -> int:
        return self.__eventstream_index

    @property
    def _hash(self) -> str:
        if self.__hash == "":
            self.__hash = hash_dataframe(self.__events)
        return self.__hash

    def __track_dataset(
        self,
        name: str,
        data: pd.DataFrame | pd.Series[Any],
        params: dict[str, Any],
        schema: RawDataSchema | RawDataSchemaType | EventstreamSchema,
        not_hash_values: list[str],
    ) -> None:
        try:
            unique_users = data[schema.user_id].nunique()
        except Exception as e:
            unique_users = None

        try:
            unique_events = data[schema.event_name].nunique()
        except Exception as e:
            unique_events = None
        try:
            hist_data = data[schema.user_id].drop_duplicates()
            if len(hist_data) >= 500:
                hist_data = hist_data.sample(500, random_state=42)
            eventstream_hist = (
                data[data[schema.user_id].isin(hist_data)].groupby(schema.user_id).size().value_counts().to_dict()
            )

        except Exception:
            eventstream_hist = {}
        eventstream_hash = hash_dataframe(data=data)
        self.__hash = eventstream_hash

        performance_data: dict[str, Any] = {
            "shape": data.shape,
            "custom_cols": len(self.schema.custom_cols),
            "unique_users": unique_users,
            "unique_events": unique_events,
            "hash": self._hash,
            "eventstream_hist": eventstream_hist,
            "index": self.__eventstream_index,
        }
        collect_data_performance(
            scope="eventstream",
            event_name=name,
            called_params=params,
            not_hash_values=not_hash_values,
            performance_data=performance_data,
            eventstream_index=self._eventstream_index,
        )

    @time_performance(
        scope="eventstream",
        event_name="copy",
    )
    def copy(self) -> Eventstream:
        """
        Make a copy of current ``eventstream``.

        Returns
        -------
        Eventstream

        """
        copied_eventstream = Eventstream(
            raw_data_schema=self.__raw_data_schema.copy(),
            raw_data=self.__events.copy(),
            schema=self.schema.copy(),
            prepare=False,
            index_order=self.index_order.copy(),
            events_order=self.events_order.copy(),
            add_start_end_events=False,
        )
        collect_data_performance(
            scope="eventstream",
            event_name="metadata",
            called_params={},
            performance_data={},
            eventstream_index=self._eventstream_index,
            parent_eventstream_index=self._eventstream_index,
            child_eventstream_index=copied_eventstream._eventstream_index,
        )
        return copied_eventstream

    @time_performance(
        scope="eventstream",
        event_name="append_eventstream",
    )
    def append_eventstream(self, eventstream: Eventstream) -> None:  # type: ignore
        """
        Append ``eventstream`` with the same schema.

        Parameters
        ----------
        eventstream : Eventstream

        Returns
        -------
        eventstream

        Raises
        ------
        ValueError
            If ``EventstreamSchemas`` of two ``eventstreams`` are not equal.
        """
        if not self.schema.is_equal(eventstream.schema):
            raise ValueError("invalid schema: joined eventstream")

        collect_data_performance(
            scope="eventstream",
            event_name="metadata",
            called_params={},
            performance_data={},
            eventstream_index=self._eventstream_index,
            parent_eventstream_index=eventstream._eventstream_index,
            child_eventstream_index=self._eventstream_index,
        )

        curr_events = self.to_dataframe()
        new_events = eventstream.to_dataframe()

        merged_events = pd.merge(
            curr_events,
            new_events,
            left_on=self.schema.event_id,
            right_on=self.schema.event_id,
            how="outer",
            indicator=True,
        )

        left_events = merged_events[(merged_events["_merge"] == "left_only")]
        both_events = merged_events[(merged_events["_merge"] == "both")]
        right_events = merged_events[(merged_events["_merge"] == "right_only")]

        right_events = pd.concat([right_events, both_events])

        cols = self.schema.get_cols()

        result_left_part = pd.DataFrame()
        result_right_part = pd.DataFrame()
        result_both_part = pd.DataFrame()

        with warnings.catch_warnings():
            # disable warning for pydantic schema Callable type
            warnings.simplefilter(action="ignore", category=FutureWarning)

            for col in cols:
                result_left_part[col] = get_merged_col(df=left_events, colname=col, suffix="_x")
                result_right_part[col] = get_merged_col(df=right_events, colname=col, suffix="_y")

        self.__events = pd.concat([result_left_part, result_both_part, result_right_part])
        self.index_events()

    def _get_both_custom_cols(self, eventstream: Eventstream) -> list[str]:
        self_custom_cols = set(self.schema.custom_cols)
        eventstream_custom_cols = set(eventstream.schema.custom_cols)
        all_custom_cols = self_custom_cols.union(eventstream_custom_cols)
        return list(all_custom_cols)

    def _get_both_cols(self, eventstream: Eventstream) -> list[str]:
        self_cols = set(self.schema.get_cols())
        eventstream_cols = set(eventstream.schema.get_cols())
        all_cols = self_cols.union(eventstream_cols)
        return list(all_cols)

    def _create_index(self, events: pd.DataFrame) -> pd.DataFrame:
        events_order_sort_col = "events_order_sort_col"
        events_type_sort_col = "events_type_sort_col"

        events[events_order_sort_col] = np.vectorize(self.__get_events_priority_by_config, otypes=[np.int64])(
            events[self.schema.event_name]
        )
        events[events_type_sort_col] = np.vectorize(self.__get_events_priority_by_type, otypes=[np.int64])(
            events[self.schema.event_type]
        )

        events = events.sort_values(
            [self.schema.event_timestamp, events_order_sort_col, events_type_sort_col]
        )  # type: ignore
        events = events.drop([events_order_sort_col, events_type_sort_col], axis=1)
        events.reset_index(inplace=True, drop=True)
        events[self.schema.event_index] = events.index
        return events

    @time_performance(
        scope="eventstream",
        event_name="to_dataframe",
    )
    def to_dataframe(self, copy: bool = False, drop_segment_events: bool = True) -> pd.DataFrame:
        """
        Convert ``eventstream`` to ``pd.DataFrame``

        Parameters
        ----------
        copy : bool, default False
            If ``True`` copy data from current eventstream.
            See details in the :pandas_copy:`pandas documentation<>`.

        drop_segment_events : bool, default True
            If ``True`` remove segment synthetic events.

        Returns
        -------
        pd.DataFrame

        """
        params: dict[str, Any] = {
            "copy": copy,
            "drop_segment_events": drop_segment_events,
        }

        schema = self.schema
        events = self.__events

        if drop_segment_events:
            events = events[events[schema.event_type] != SEGMENT_TYPE]

        view = pd.DataFrame(events, columns=self.schema.get_cols(), copy=copy)
        self.__track_dataset(name="metadata", data=view, params=params, schema=self.schema, not_hash_values=[])
        return view

    @time_performance(
        scope="eventstream",
        event_name="index_events",
    )
    def index_events(self) -> None:
        """
        Sort and index eventstream using DEFAULT_INDEX_ORDER.

        Returns
        -------
        None

        """
        collect_data_performance(
            scope="eventstream",
            event_name="metadata",
            called_params={},
            performance_data={},
            eventstream_index=self._eventstream_index,
        )
        order_temp_col_name = "order"
        indexed = self.__events

        indexed[order_temp_col_name] = np.vectorize(self.__get_events_priority_by_type, otypes=[np.int64])(
            indexed[self.schema.event_type]
        )
        indexed = indexed.sort_values(
            [self.schema.event_timestamp, self.schema.event_index, order_temp_col_name]
        )  # type: ignore
        indexed = indexed.drop([order_temp_col_name], axis=1)
        indexed.reset_index(inplace=True, drop=True)
        self.__events = indexed

    @time_performance(
        scope="eventstream",
        event_name="add_custom_col",
    )
    def add_custom_col(self, name: str, data: pd.Series[Any] | None) -> None:
        """
        Add custom column to an existing ``eventstream``.

        Parameters
        ----------
        name : str
            New column name.
        data : pd.Series

            - If ``pd.Series`` - new column with given values will be added.
            - If ``None`` - new column will be filled with ``np.nan``.

        Returns
        -------
        Eventstream
        """
        collect_data_performance(
            scope="eventstream",
            event_name="metadata",
            called_params={"name": name, "data": data},
            performance_data={},
            eventstream_index=self._eventstream_index,
        )

        if name not in self.schema.custom_cols:
            self.__raw_data_schema.custom_cols.extend([{"custom_col": name, "raw_data_col": name}])
            self.schema.custom_cols.extend([name])
        self.__events[name] = data

    def __define_default_custom_cols(self, raw_data: pd.DataFrame | pd.Series[Any]) -> List[str]:
        raw_data_cols = self.__raw_data_schema.get_default_cols()
        schema_cols = self.schema.get_default_cols()

        cols_denylist: List[str] = raw_data_cols + schema_cols

        custom_cols: List[str] = []

        for raw_col_name in raw_data.columns:
            if raw_col_name in cols_denylist:
                continue
            custom_cols.append(raw_col_name)

        return custom_cols

    def __required_cleanup(self, events: pd.DataFrame | pd.Series[Any]) -> pd.DataFrame | pd.Series[Any]:
        income_size = len(events)
        events.dropna(  # type: ignore
            subset=[self.schema.event_name, self.schema.event_timestamp, self.schema.user_id], inplace=True
        )
        size_after_cleanup = len(events)
        if (removed_rows := income_size - size_after_cleanup) > 0:
            warnings.warn(
                "Removed %s rows because they have empty %s or %s or %s"
                % (removed_rows, self.schema.event_name, self.schema.event_timestamp, self.schema.user_id)
            )
        return events

    def __prepare_events(self, raw_data: pd.DataFrame | pd.Series[Any]) -> pd.DataFrame | pd.Series[Any]:
        events = pd.DataFrame(index=raw_data.index)

        if self.__raw_data_schema.event_id is not None and self.__raw_data_schema.event_id in raw_data.columns:
            events[self.schema.event_id] = raw_data[self.__raw_data_schema.event_id]
        else:
            events[self.schema.event_id] = [uuid.uuid4() for x in range(len(events))]

        events[self.schema.event_name] = self.__get_col_from_raw_data(
            raw_data=raw_data,
            colname=self.__raw_data_schema.event_name,
        )

        events[self.schema.event_timestamp] = self.__get_col_from_raw_data(
            raw_data=raw_data, colname=self.__raw_data_schema.event_timestamp
        )

        events[self.schema.event_timestamp] = pd.to_datetime(events[self.schema.event_timestamp])

        has_tz = self.convert_tz is not None
        if not has_tz:
            try:
                has_tz = events[self.schema.event_timestamp].dt.tz is not None
            except AttributeError:
                has_tz = any(date.tzinfo for date in events[self.schema.event_timestamp])

        if has_tz:
            if not self.convert_tz:
                error_text = (
                    "Eventstream doesn't support working with the timestamps that contain timezone data.\n"
                    "Use convert_tz='UTC' or convert_tz='local' arguments to convert the timezones explicitly."
                )
                raise TypeError(error_text)
            elif self.convert_tz == "UTC":
                events[self.schema.event_timestamp] = pd.to_datetime(
                    events[self.schema.event_timestamp], utc=True
                ).dt.tz_localize(None)
            elif self.convert_tz == "local":
                # @TODO: regexp is potentially dangerous since it may not cover all possible data formats.
                #  It'd be better to find more native yet vectorized way to do this. Vladimir Kukushkin
                tz_pattern = r"[+-]\d{2}:\d{2}"
                cut_timezones = events[self.schema.event_timestamp].astype(str).str.replace(tz_pattern, "", regex=True)
                events[self.schema.event_timestamp] = pd.to_datetime(cut_timezones)
            else:
                raise TypeError(f"convert_tz parameter should be either None, or 'UTC', or 'local'")

        events[self.schema.user_id] = self.__get_col_from_raw_data(
            raw_data=raw_data,
            colname=self.__raw_data_schema.user_id,
        )

        if self.__raw_data_schema.event_type is not None:
            events[self.schema.event_type] = self.__get_col_from_raw_data(
                raw_data=raw_data,
                colname=self.__raw_data_schema.event_type,
            )
        else:
            events[self.schema.event_type] = "raw"

        for custom_col_schema in self.__raw_data_schema.custom_cols:
            raw_data_col = custom_col_schema["raw_data_col"]
            custom_col = custom_col_schema["custom_col"]
            if custom_col not in self.schema.custom_cols:
                self.schema.custom_cols.append(custom_col)

            events[custom_col] = self.__get_col_from_raw_data(
                raw_data=raw_data,
                colname=raw_data_col,
            )

        for custom_col in self.schema.custom_cols:
            if custom_col in events.columns:
                continue
            events[custom_col] = np.nan

        if self.__raw_data_schema.event_index is not None and self.__raw_data_schema.event_index in raw_data.columns:
            events[self.schema.event_index] = raw_data[self.__raw_data_schema.event_index].astype("int64")
        else:
            events = self._create_index(events=events)  # type: ignore

        return events

    def __apply_default_dataprocessors(
        self, add_start_end_events: bool, segment_cols: Optional[List[str]] = None
    ) -> None:
        from retentioneering.data_processors_lib.add_segment import (
            AddSegment,
            AddSegmentParams,
        )
        from retentioneering.data_processors_lib.add_start_end_events import (
            AddStartEndEvents,
            AddStartEndEventsParams,
        )

        if add_start_end_events:
            add_start_end_processor = AddStartEndEvents(AddStartEndEventsParams())
            self.__events = add_start_end_processor.apply(self.__events, self.schema)  # type: ignore

        if segment_cols:
            for segment_col in segment_cols:
                add_segment_processor = AddSegment(AddSegmentParams(segment=segment_col))  # type: ignore
                self.__events = add_segment_processor.apply(self.__events, self.schema)  # type: ignore

    def __get_col_from_raw_data(
        self, raw_data: pd.DataFrame | pd.Series[Any], colname: str, create: bool = False
    ) -> pd.Series | float:
        if colname in raw_data.columns:
            return raw_data[colname]
        else:
            if create:
                return np.nan
            else:
                raise ValueError(f'invalid raw data. Column "{colname}" does not exists!')

    def __get_events_priority_by_type(self, event_type: str) -> int:
        return self.index_order.get(event_type, len(self.index_order))

    def __get_events_priority_by_config(self, event_name: str) -> int:
        return self.events_order.get(event_name, len(self.events_order))

    def __sample_user_paths(
        self,
        raw_data: pd.DataFrame | pd.Series[Any],
        raw_data_schema: RawDataSchemaType,
        user_sample_size: Optional[int | float] = None,
        user_sample_seed: Optional[int] = None,
    ) -> pd.DataFrame | pd.Series[Any]:
        if type(user_sample_size) is not float and type(user_sample_size) is not int:
            raise TypeError('"user_sample_size" has to be a number(float for user share or int for user amount)')
        if user_sample_size < 0:
            raise ValueError("User sample size/share cannot be negative!")
        if type(user_sample_size) is float:
            if user_sample_size > 1:
                raise ValueError("User sample share cannot exceed 1!")
        user_col_name = raw_data_schema.user_id
        unique_users = raw_data[user_col_name].unique()
        if type(user_sample_size) is int:
            sample_size = user_sample_size
        elif type(user_sample_size) is float:
            sample_size = int(user_sample_size * len(unique_users))
        else:
            return raw_data
        if user_sample_seed is not None:
            np.random.seed(user_sample_seed)
        sample_users = np.random.choice(unique_users, sample_size, replace=False)
        raw_data_sampled = raw_data.loc[raw_data[user_col_name].isin(sample_users), :]  # type: ignore
        return raw_data_sampled

    @time_performance(
        scope="funnel",
        event_name="helper",
        event_value="plot",
    )
    def funnel(
        self,
        stages: list[str],
        stage_names: list[str] | None = None,
        funnel_type: Literal["open", "closed", "hybrid"] = "closed",
        groups: SplitExpr | None = None,
        group_names: UserGroupsNamesType | None = None,
        show_plot: bool = True,
    ) -> Funnel:
        """
        Show a visualization of the user sequential events represented as a funnel.

        Parameters
        ----------
        show_plot : bool, default True
            If ``True``, a funnel visualization is shown.
        See other parameters' description
            :py:class:`.Funnel`

        Returns
        -------
        Funnel
            A ``Funnel`` class instance fitted to the given parameters.

        """
        params = {
            "stages": stages,
            "stage_names": stage_names,
            "funnel_type": funnel_type,
            "groups": groups,
            "group_names": group_names,
            "show_plot": show_plot,
        }

        collect_data_performance(
            scope="funnel",
            event_name="metadata",
            called_params=params,
            not_hash_values=["funnel_type"],
            performance_data={},
            eventstream_index=self._eventstream_index,
        )
        self.__funnel = Funnel(eventstream=self)
        self.__funnel.fit(
            stages=stages,
            stage_names=stage_names,
            funnel_type=funnel_type,
            groups=groups,
            group_names=group_names,
        )
        if show_plot:
            figure = self.__funnel.plot()
            figure.show()
        return self.__funnel

    def get_clusters(
        self,
        X: pd.DataFrame,
        method: CLUSTERING_METHODS,
        n_clusters: int | None = None,
        scaler: SCALERS | None = None,
        random_state: int | None = None,
        segment_name: str = "cluster_id",
        **kwargs: Any,
    ) -> EventstreamType | None:
        """
        Split paths into clusters and save their labels as a segment.

        Parameters
        ----------

        X : pd.DataFrame
            The input data to cluster.

        method : {"kmeans", "gmm", "hdbscan"}
            The clustering method to use.

        n_clusters : int, optional
            The number of clusters to form. Actual for ``kmeans`` and ``gmm`` methods.
            If ``n_clusters=None`` and ``method="kmeans"``, the elbow curve chart is displayed.

        scaler : {"minmax", "std"}, optional
            The scaling method to apply to the data before clustering. If None, no scaling is applied.

        random_state : int, optional
            A seed used by the random number generator for reproducibility.

        segment_name : str, default "cluster_id"
            The name of the segment that will contain the cluster labels.

        **kwargs
            Additional keyword arguments to pass to the clustering methods.

        Returns
        -------

        Eventstream or None
            If ``n_clusters`` is specified, a new Eventstream object with clusters integrated as a segment
            ``segment_name`` is returned; otherwise, returns ``None``.
        """
        self.__clusters = Clusters(eventstream=self)
        new_stream = self.__clusters.fit(
            X=X,
            method=method,
            n_clusters=n_clusters,
            scaler=scaler,
            random_state=random_state,
            segment_name=segment_name,
            **kwargs,
        )
        if n_clusters is not None:
            return new_stream
        return self

    @time_performance(
        scope="step_matrix",
        event_name="helper",
        event_value="plot",
    )
    def step_matrix(
        self,
        max_steps: int = 20,
        weight_col: str | None = None,
        precision: int = 2,
        targets: list[str] | str | None = None,
        accumulated: Literal["both", "only"] | None = None,
        sorting: list | None = None,
        threshold: float = 0.01,
        centered: dict | None = None,
        groups: SplitExpr | None = None,
        show_plot: bool = True,
    ) -> StepMatrix:
        """
        Show a heatmap visualization of the step matrix.

        Parameters
        ----------
        show_plot : bool, default True
            If ``True``, a step matrix heatmap is shown.
        See other parameters' description
            :py:class:`.StepMatrix`

        Returns
        -------
        StepMatrix
            A ``StepMatrix`` class instance fitted to the given parameters.

        """

        params = {
            "max_steps": max_steps,
            "weight_col": weight_col,
            "precision": precision,
            "targets": targets,
            "accumulated": accumulated,
            "sorting": sorting,
            "threshold": threshold,
            "centered": centered,
            "groups": groups,
            "show_plot": show_plot,
        }
        not_hash_values = ["accumulated", "centered"]
        collect_data_performance(
            scope="step_matrix",
            event_name="metadata",
            called_params=params,
            not_hash_values=not_hash_values,
            performance_data={},
            eventstream_index=self._eventstream_index,
        )
        self.__step_matrix = StepMatrix(eventstream=self)

        self.__step_matrix.fit(
            max_steps=max_steps,
            weight_col=weight_col,
            precision=precision,
            targets=targets,
            accumulated=accumulated,
            sorting=sorting,
            threshold=threshold,
            centered=centered,
            groups=groups,
        )
        if show_plot:
            self.__step_matrix.plot()
        return self.__step_matrix

    @time_performance(
        scope="step_sankey",
        event_name="helper",
        event_value="plot",
    )
    def step_sankey(
        self,
        max_steps: int = 10,
        threshold: int | float = 0.05,
        sorting: list | None = None,
        targets: list[str] | str | None = None,
        autosize: bool = True,
        width: int | None = None,
        height: int | None = None,
        show_plot: bool = True,
    ) -> StepSankey:
        """
        Show a Sankey diagram visualizing the user paths in stepwise manner.

        Parameters
        ----------
        show_plot : bool, default True
            If ``True``, a sankey diagram is shown.
        See other parameters' description
            :py:class:`.StepSankey`

        Returns
        -------
        StepSankey
            A ``StepSankey`` class instance fitted to the given parameters.

        """

        params = {
            "max_steps": max_steps,
            "threshold": threshold,
            "sorting": sorting,
            "targets": targets,
            "autosize": autosize,
            "width": width,
            "height": height,
            "show_plot": show_plot,
        }

        collect_data_performance(
            scope="step_sankey",
            event_name="metadata",
            called_params=params,
            performance_data={},
            eventstream_index=self._eventstream_index,
        )

        self.__sankey = StepSankey(eventstream=self)

        self.__sankey.fit(max_steps=max_steps, threshold=threshold, sorting=sorting, targets=targets)
        if show_plot:
            figure = self.__sankey.plot(autosize=autosize, width=width, height=height)
            figure.show()
        return self.__sankey

    @time_performance(
        scope="cohorts",
        event_name="helper",
        event_value="heatmap",
    )
    def cohorts(
        self,
        cohort_start_unit: DATETIME_UNITS,
        cohort_period: Tuple[int, DATETIME_UNITS],
        average: bool = True,
        cut_bottom: int = 0,
        cut_right: int = 0,
        cut_diagonal: int = 0,
        width: float = 5.0,
        height: float = 5.0,
        show_plot: bool = True,
    ) -> Cohorts:
        """
        Show a heatmap visualization of the user appearance grouped by cohorts.

        Parameters
        ----------
        show_plot : bool, default True
            If ``True``, a cohort matrix heatmap is shown.
        See other parameters' description
            :py:class:`.Cohorts`

        Returns
        -------
        Cohorts
            A ``Cohorts`` class instance fitted to the given parameters.
        """

        params = {
            "cohort_start_unit": cohort_start_unit,
            "cohort_period": cohort_period,
            "average": average,
            "cut_bottom": cut_bottom,
            "cut_right": cut_right,
            "cut_diagonal": cut_diagonal,
            "width": width,
            "height": height,
            "show_plot": show_plot,
        }

        not_hash_values = ["cohort_start_unit", "cohort_period"]
        collect_data_performance(
            scope="cohorts",
            event_name="metadata",
            called_params=params,
            not_hash_values=not_hash_values,
            performance_data={},
            eventstream_index=self._eventstream_index,
        )

        self.__cohorts = Cohorts(eventstream=self)

        self.__cohorts.fit(
            cohort_start_unit=cohort_start_unit,
            cohort_period=cohort_period,
            average=average,
            cut_bottom=cut_bottom,
            cut_right=cut_right,
            cut_diagonal=cut_diagonal,
        )
        if show_plot:
            self.__cohorts.heatmap(width=width, height=height)
        return self.__cohorts

    @time_performance(
        scope="stattests",
        event_name="helper",
        event_value="display_results",
    )
    def stattests(
        self,
        test: STATTEST_NAMES,
        groups: SplitExpr,
        func: Callable,
        group_names: UserGroupsNamesType,
        alpha: float = 0.05,
    ) -> StatTests:
        """
        Determine the statistical difference between the metric values in two user groups.

        Parameters
        ----------
        See parameters' description
            :py:class:`.Stattests`

        Returns
        -------
        StatTests
            A ``StatTest`` class instance fitted to the given parameters.
        """
        params = {
            "test": test,
            "groups": groups,
            "func": func,
            "group_names": group_names,
            "alpha": alpha,
        }
        not_hash_values = ["test"]

        collect_data_performance(
            scope="stattests",
            event_name="metadata",
            called_params=params,
            not_hash_values=not_hash_values,
            performance_data={},
            eventstream_index=self._eventstream_index,
        )

        self.__stattests = StatTests(eventstream=self)
        self.__stattests.fit(groups=groups, func=func, test=test, group_names=group_names, alpha=alpha)
        self.__stattests.display_results()
        return self.__stattests

    @time_performance(
        scope="timedelta_hist",
        event_name="helper",
        event_value="plot",
    )
    def timedelta_hist(
        self,
        raw_events_only: bool = False,
        event_pair: list[str | Literal[EVENTSTREAM_GLOBAL_EVENTS]] | None = None,
        adjacent_events_only: bool = True,
        weight_col: str | None = None,
        time_agg: AGGREGATION_NAMES | None = None,
        timedelta_unit: DATETIME_UNITS = "s",
        log_scale: bool | tuple[bool, bool] | None = None,
        lower_cutoff_quantile: float | None = None,
        upper_cutoff_quantile: float | None = None,
        bins: int | Literal[BINS_ESTIMATORS] = 20,
        width: float = 6.0,
        height: float = 4.5,
        show_plot: bool = True,
    ) -> TimedeltaHist:
        """
        Plot the distribution of the time deltas between two events. Support various
        distribution types, such as distribution of time for adjacent consecutive events, or
        for a pair of pre-defined events, or median transition time from event to event per user/session.

        Parameters
        ----------
        show_plot : bool, default True
            If ``True``, histogram is shown.
        See other parameters' description
            :py:class:`.TimedeltaHist`

        Returns
        -------
        TimedeltaHist
            A ``TimedeltaHist`` class instance fitted with given parameters.

        """

        params = {
            "raw_events_only": raw_events_only,
            "event_pair": event_pair,
            "adjacent_events_only": adjacent_events_only,
            "weight_col": weight_col,
            "time_agg": time_agg,
            "timedelta_unit": timedelta_unit,
            "log_scale": log_scale,
            "lower_cutoff_quantile": lower_cutoff_quantile,
            "upper_cutoff_quantile": upper_cutoff_quantile,
            "bins": bins,
            "width": width,
            "height": height,
            "show_plot": show_plot,
        }
        not_hash_values = ["time_agg", "timedelta_unit"]

        collect_data_performance(
            scope="timedelta_hist",
            event_name="metadata",
            called_params=params,
            not_hash_values=not_hash_values,
            performance_data={},
            eventstream_index=self._eventstream_index,
        )

        self.__timedelta_hist = TimedeltaHist(
            eventstream=self,
        )

        self.__timedelta_hist.fit(
            raw_events_only=raw_events_only,
            event_pair=event_pair,
            adjacent_events_only=adjacent_events_only,
            time_agg=time_agg,
            weight_col=weight_col,
            timedelta_unit=timedelta_unit,
            log_scale=log_scale,
            lower_cutoff_quantile=lower_cutoff_quantile,
            upper_cutoff_quantile=upper_cutoff_quantile,
            bins=bins,
        )
        if show_plot:
            self.__timedelta_hist.plot(
                width=width,
                height=height,
            )

        return self.__timedelta_hist

    @time_performance(
        scope="user_lifetime_hist",
        event_name="helper",
        event_value="plot",
    )
    def user_lifetime_hist(
        self,
        timedelta_unit: DATETIME_UNITS = "s",
        log_scale: bool | tuple[bool, bool] | None = None,
        lower_cutoff_quantile: float | None = None,
        upper_cutoff_quantile: float | None = None,
        bins: int | Literal[BINS_ESTIMATORS] = 20,
        width: float = 6.0,
        height: float = 4.5,
        show_plot: bool = True,
    ) -> UserLifetimeHist:
        """
        Plot the distribution of user lifetimes. A ``users lifetime`` is the timedelta between the first and the last
        events of the user.

        Parameters
        ----------
        show_plot : bool, default True
            If ``True``, histogram is shown.
        See other parameters' description
            :py:class:`.UserLifetimeHist`

        Returns
        -------
        UserLifetimeHist
            A ``UserLifetimeHist`` class instance with given parameters.


        """
        params = {
            "timedelta_unit": timedelta_unit,
            "log_scale": log_scale,
            "lower_cutoff_quantile": lower_cutoff_quantile,
            "upper_cutoff_quantile": upper_cutoff_quantile,
            "bins": bins,
            "width": width,
            "height": height,
            "show_plot": show_plot,
        }
        not_hash_values = ["timedelta_unit"]

        collect_data_performance(
            scope="user_lifetime_hist",
            event_name="metadata",
            called_params=params,
            not_hash_values=not_hash_values,
            performance_data={},
            eventstream_index=self._eventstream_index,
        )
        self.__user_lifetime_hist = UserLifetimeHist(
            eventstream=self,
        )
        self.__user_lifetime_hist.fit(
            timedelta_unit=timedelta_unit,
            log_scale=log_scale,
            lower_cutoff_quantile=lower_cutoff_quantile,
            upper_cutoff_quantile=upper_cutoff_quantile,
            bins=bins,
        )
        if show_plot:
            self.__user_lifetime_hist.plot(width=width, height=height)
        return self.__user_lifetime_hist

    @time_performance(
        scope="event_timestamp_hist",
        event_name="helper",
        event_value="plot",
    )
    def event_timestamp_hist(
        self,
        event_list: list[str] | None = None,
        raw_events_only: bool = False,
        lower_cutoff_quantile: float | None = None,
        upper_cutoff_quantile: float | None = None,
        bins: int | Literal[BINS_ESTIMATORS] = 20,
        width: float = 6.0,
        height: float = 4.5,
        show_plot: bool = True,
    ) -> EventTimestampHist:
        """
        Plot distribution of events over time. Can be useful for detecting time-based anomalies, and visualising
        general timespan of the eventstream.

        Parameters
        ----------
        show_plot : bool, default True
            If ``True``, histogram is shown.
        See other parameters' description
            :py:class:`.EventTimestampHist`


        Returns
        -------
        EventTimestampHist
            A ``EventTimestampHist`` class instance with given parameters.
        """
        params = {
            "event_list": event_list,
            "raw_events_only": raw_events_only,
            "lower_cutoff_quantile": lower_cutoff_quantile,
            "upper_cutoff_quantile": upper_cutoff_quantile,
            "bins": bins,
            "width": width,
            "height": height,
            "show_plot": show_plot,
        }

        collect_data_performance(
            scope="event_timestamp_hist",
            event_name="metadata",
            called_params=params,
            performance_data={},
            eventstream_index=self._eventstream_index,
        )
        self.__event_timestamp_hist = EventTimestampHist(
            eventstream=self,
        )

        self.__event_timestamp_hist.fit(
            event_list=event_list,
            raw_events_only=raw_events_only,
            lower_cutoff_quantile=lower_cutoff_quantile,
            upper_cutoff_quantile=upper_cutoff_quantile,
            bins=bins,
        )
        if show_plot:
            self.__event_timestamp_hist.plot(width=width, height=height)
        return self.__event_timestamp_hist

    @time_performance(
        scope="describe",
        event_name="helper",
        event_value="_values",
    )
    def describe(self, session_col: str = "session_id", raw_events_only: bool = False) -> pd.DataFrame:
        """
        Display general eventstream information. If ``session_col`` is present in eventstream, also
        output session statistics.

        Parameters
        ----------
        session_col : str, default 'session_id'
            Specify name of the session column. If the column is present in the eventstream,
            session statistics will be added to the output.

        raw_events_only : bool, default False
            If ``True`` - statistics will only be shown for raw events.
            If ``False`` - statistics will be shown for all events presented in your data.

        Returns
        -------
        pd.DataFrame
            A dataframe containing descriptive statistics for the eventstream.


        See Also
        --------
        .EventTimestampHist : Plot the distribution of events over time.
        .TimedeltaHist : Plot the distribution of the time deltas between two events.
        .UserLifetimeHist : Plot the distribution of user lifetimes.
        .Eventstream.describe_events : Show general eventstream events statistics.


        Notes
        -----
        - All ``float`` values are rounded to 2.
        - All ``datetime`` values are rounded to seconds.

        See :ref:`Eventstream user guide<eventstream_describe>` for the details.


        """
        params = {
            "session_col": session_col,
            "raw_events_only": raw_events_only,
        }

        collect_data_performance(
            scope="describe",
            event_name="metadata",
            called_params=params,
            performance_data={},
            eventstream_index=self._eventstream_index,
        )
        describer = _Describe(eventstream=self, session_col=session_col, raw_events_only=raw_events_only)
        return describer._values()

    @time_performance(
        scope="describe_events",
        event_name="helper",
        event_value="_values",
    )
    def describe_events(
        self, session_col: str = "session_id", raw_events_only: bool = False, event_list: list[str] | None = None
    ) -> pd.DataFrame:
        """
        Display general information on eventstream events. If ``session_col`` is present in eventstream, also
        output session statistics.

        Parameters
        ----------
        session_col : str, default 'session_id'
            Specify name of the session column. If the column is present in the eventstream,
            output session statistics.

        raw_events_only : bool, default False
            If ``True`` - statistics will only be shown for raw events.
            If ``False`` - statistics will be shown for all events presented in your data.

        event_list : list of str, optional
            Specify events to be displayed.

        Returns
        -------
        pd.DataFrame
            **Eventstream statistics**:

            - The following metrics are calculated for each event present in the eventstream
              (or the narrowed eventstream if parameters ``event_list`` or ``raw_events_only`` are used).
              Let all_events, all_users, all_sessions be the numbers of all events, users,
              and sessions present in the eventstream. Then:

                - *number_of_occurrences* - the number of occurrences of a particular event in the eventstream;
                - *unique_users* - the number of unique users who experienced a particular event;
                - *unique_sessions* - the number of unique sessions with each event;
                - *number_of_occurrences_shared* - number_of_occurrences / all_events (raw_events_only,
                  if this parameter = ``True``);
                - *unique_users_shared* - unique_users / all_users;
                - *unique_sessions_shared* - unique_sessions / all_sessions;

            - **time_to_FO_user_wise** category - timedelta between ``path_start``
              and the first occurrence (FO) of a specified event in each user path.
            - **steps_to_FO_user_wise** category - the number of steps (events) from
              ``path_start`` to the first occurrence (FO) of a specified event in each user path.
              If ``raw_events_only=True`` only raw events will be counted.
            - **time_to_FO_session_wise** category - timedelta  between ``session_start``
              and the first occurrence (FO) of a specified event in each session.
            - **steps_to_FO_session_wise** category - the number of steps (events) from
              ``session_start`` to the first occurrence (FO) of a specified event in each session.
              If ``raw_events_only=True`` only raw events will be counted.

            Agg functions for each ``first_occurrence*`` category are: mean, std, median, min, max.

        See Also
        --------
        .EventTimestampHist : Plot the distribution of events over time.
        .TimedeltaHist : Plot the distribution of the time deltas between two events.
        .UserLifetimeHist : Plot the distribution of user lifetimes.
        .Eventstream.describe : Show general eventstream statistics.

        Notes
        -----
        - All ``float`` values are rounded to 2.
        - All ``datetime`` values are rounded to seconds.

        See :ref:`Eventstream user guide<eventstream_describe_events>` for the details.

        """

        params = {
            "session_col": session_col,
            "raw_events_only": raw_events_only,
            "event_list": event_list,
        }

        collect_data_performance(
            scope="describe_events",
            event_name="metadata",
            called_params=params,
            performance_data={},
            eventstream_index=self._eventstream_index,
        )

        describer = _DescribeEvents(
            eventstream=self, session_col=session_col, event_list=event_list, raw_events_only=raw_events_only
        )
        return describer._values()

    @time_performance(
        scope="transition_graph",
        event_name="helper",
        event_value="plot",
    )
    def transition_graph(
        self,
        edges_norm_type: NormType = None,
        nodes_norm_type: NormType = None,
        targets: MutableMapping[str, str | None] | None = None,
        nodes_threshold: Threshold | None = None,
        edges_threshold: Threshold | None = None,
        nodes_weight_col: str | None = None,
        edges_weight_col: str | None = None,
        custom_weight_cols: list[str] | None = None,
        width: str | int | float = "100%",
        height: str | int | float = "60vh",
        show_weights: bool = True,
        show_percents: bool = False,
        show_nodes_names: bool = True,
        show_all_edges_for_targets: bool = True,
        show_nodes_without_links: bool = False,
        show_edge_info_on_hover: bool = True,
        nodes_custom_colors: Dict[str, str] | None = None,
        edges_custom_colors: Dict[Tuple[str, str], str] | None = None,
        layout_file: str | None = None,
        nodelist: Nodelist | pd.DataFrame | None = None,
        import_file: str | None = None,
    ) -> TransitionGraph:
        """

        Parameters
        ----------
        See parameters' description
            :py:meth:`.TransitionGraph.plot`
            @TODO: maybe load docs with docrep? 2dpanina, Vladimir Makhanov

        Returns
        -------
        TransitionGraph
            Rendered IFrame graph.

        """

        params = {
            "edges_norm_type": edges_norm_type,
            "nodes_norm_type": nodes_norm_type,
            "targets": targets,
            "nodes_threshold": nodes_threshold,
            "edges_threshold": edges_threshold,
            "nodes_weight_col": nodes_weight_col,
            "edges_weight_col": edges_weight_col,
            "custom_weight_cols": custom_weight_cols,
            "width": width,
            "height": height,
            "show_weights": show_weights,
            "show_percents": show_percents,
            "show_nodes_names": show_nodes_names,
            "show_all_edges_for_targets": show_all_edges_for_targets,
            "show_nodes_without_links": show_nodes_without_links,
            "show_edge_info_on_hover": show_edge_info_on_hover,
            "nodes_custom_colors": nodes_custom_colors,
            "edges_custom_colors": edges_custom_colors,
            "layout_file": layout_file,
            "nodelist": nodelist,
            "import_file": import_file,
        }
        not_hash_values = ["edges_norm_type", "targets", "width", "height", "edges_custom_colors"]

        collect_data_performance(
            scope="transition_graph",
            event_name="metadata",
            called_params=params,
            not_hash_values=not_hash_values,
            performance_data={},
            eventstream_index=self._eventstream_index,
        )

        self.__transition_graph = TransitionGraph(eventstream=self)
        self.__transition_graph.plot(
            targets=targets,
            edges_norm_type=edges_norm_type,
            nodes_norm_type=nodes_norm_type,
            edges_weight_col=edges_weight_col,
            nodes_threshold=nodes_threshold,
            edges_threshold=edges_threshold,
            nodes_weight_col=nodes_weight_col,
            custom_weight_cols=custom_weight_cols,
            width=width,
            height=height,
            show_weights=show_weights,
            show_percents=show_percents,
            show_nodes_names=show_nodes_names,
            show_all_edges_for_targets=show_all_edges_for_targets,
            show_nodes_without_links=show_nodes_without_links,
            layout_file=layout_file,
            show_edge_info_on_hover=show_edge_info_on_hover,
            nodes_custom_colors=nodes_custom_colors,
            edges_custom_colors=edges_custom_colors,
            nodelist=nodelist,
            import_file=import_file,
        )
        return self.__transition_graph

    @time_performance(
        scope="preprocessing_graph",
        event_name="helper",
        event_value="display",
    )
    def preprocessing_graph(self, width: int = 960, height: int = 600) -> PreprocessingGraph:
        """
        Display the preprocessing GUI tool.

        Parameters
        ----------
        width : int, default 960
            Width of plot in pixels.
        height : int, default 600
            Height of plot in pixels.

        Returns
        -------
        PreprocessingGraph
            Rendered preprocessing graph.
        """

        params = {
            "width": width,
            "height": height,
        }

        collect_data_performance(
            scope="preprocessing_graph",
            event_name="metadata",
            called_params=params,
            performance_data={},
            eventstream_index=self._eventstream_index,
        )

        if self._preprocessing_graph is None:
            self._preprocessing_graph = PreprocessingGraph(source_stream=self)
        self._preprocessing_graph.display(width=width, height=height)

        return self._preprocessing_graph

    @time_performance(
        scope="transition_matrix",
        event_name="helper",
        event_value="plot",
    )
    def transition_matrix(
        self,
        norm_type: NormType = None,
        weight_col: Optional[str] = None,
        groups: SplitExpr | None = None,
        heatmap_axis: Union[Literal["rows", "columns", "both"], int] = "both",
        precision: Union[int, Literal["auto"]] = "auto",
        figsize: Optional[Tuple[Union[float, int], Union[float, int]]] = None,
        show_large_matrix: Optional[bool] = None,
        show_values: Optional[bool] = None,
        show_plot: bool = True,
    ) -> TransitionMatrix:
        """
        Retrieve a matrix of transition weights for each pair of unique events.
        This function calculates transition weights based on the same logic used for calculating
        edge weights in a transition graph.

        Parameters
        ----------
        norm_type : {"full", "node", None}, default None
            Type of normalization that is used to calculate weights.
            Based on ``weight_col`` parameter the weight values are calculated.

            - If ``None``, normalization is not used, the absolute values are taken.
            - If ``full``, normalization across the whole eventstream.
            - If ``node``, normalization across each node (or outgoing transitions from each node).

            See :ref:`Transition graph user guide <transition_graph_weights>` for the details.

        weight_col : str, optional
            A column name from the :py:class:`.EventstreamSchema` which values will control the final
            edges' weights.

            For each edge is calculated:

            - If ``None`` or ``user_id`` - the number of unique users.
            - If ``event_id`` - the number of transitions.
            - If ``session_id`` - the number of unique sessions.
            - If ``custom_col`` - the number of unique values in selected column.

            See :ref:`Transition graph user guide <transition_graph_weights>` for the details.

        groups : tuple[list, list], optional
            Can be specified to calculate differential transition matrix. Must contain
            a tuple of two elements (g_1, g_2): where g_1 and g_2 are collections
            of user_id`s. Two separate transition matrices M1 and M2 will be calculated
            for users from g_1 and g_2, respectively. Resulting matrix will be the matrix
            M = M1 - M2.

        heatmap_axis : {0 or 'rows', 1 or 'columns', 'both'}, default 'both'
            The axis for which the heatmap is to be generated.
            If specified, the heatmap will be created separately for the selected axis.
            If ``heatmap_axis='both'``, the heatmap will be applied to the entire matrix.

        precision : int or str, default 'auto'
            The number of decimal digits to display after zero as fractions in the heatmap.
            If precision is ``auto``, the value will depend on the ``norm_type``:
            0 for ``norm_type=None``, and 2 otherwise.

        figsize : tuple[float, float], default None
            The size of the visualization. The default size is calculated automatically depending
            on the matrix dimension and `precision` and `show_values` options.

        show_large_matrix : bool, optional
            If ``None`` the matrix is displayed only in case the matrix dimension <= 60.
            If ``True``, the matrix is plotted explicitly.

        show_values : bool, optional
            If ``None`` the matrix values are not displayed only in case the matrix dimension lies between 30 and 60.
            If ``True``, the matrix values are shown explicitly.
            If ``False``, the values are hidden, ``precision`` parameter is ignored in this case.

        show_plot : bool, default True
            If ``True``, a heatmap of the transition matrix will be displayed.

        Returns
        -------
        TransitionMatrix
            A `TransitionMatrix` instance fitted to the given parameters is returned.

        """
        not_hash_values = ["norm_type", "heatmap_axis", "precision", "figsize", "show_large_matrix", "show_values"]

        params = {"weight_col": weight_col, "norm_type": norm_type}
        collect_data_performance(
            scope="transition_matrix",
            event_name="metadata",
            called_params=params,
            not_hash_values=not_hash_values,
            performance_data={},
            eventstream_index=self._eventstream_index,
        )

        self.__transition_matrix = TransitionMatrix(eventstream=self)
        self.__transition_matrix.fit(weight_col=weight_col, norm_type=norm_type, groups=groups)

        if show_plot:
            self.__transition_matrix.plot(
                heatmap_axis=heatmap_axis,
                precision=precision,
                figsize=figsize,
                show_large_matrix=show_large_matrix,
                show_values=show_values,
            )

        return self.__transition_matrix

    @time_performance(
        scope="sequences",
        event_name="helper",
        event_value="plot",
    )
    def sequences(
        self,
        ngram_range: Tuple[int, int] = (1, 1),
        groups: SplitExpr | None = None,
        group_names: UserGroupsNamesType | None = None,
        weight_col: str | None = None,
        metrics: SEQUENCES_METRIC_TYPES | None = None,
        threshold: tuple[str, float | int] | None = None,
        sorting: tuple[str | tuple, bool] | tuple[list[str | tuple], list[bool]] | None = None,
        heatmap_cols: str | list[str | tuple] | None = None,
        sample_size: int | None = 1,
        precision: int = 2,
        show_plot: bool = True,
    ) -> Sequences:
        """
        Calculate statistics on n-grams found in eventstream.

        Parameters
        ----------
        show_plot : bool, default True
            If ``True``, a sankey diagram is shown.
        See other parameters' description
            :py:class:`.Sequences`

        Returns
        -------
        Sequences
            A ``Sequences`` class instance fitted to the given parameters.

        """

        params = {
            "ngram_range": ngram_range,
            "groups": groups,
            "group_names": group_names,
            "weight_col": weight_col,
            "metrics": metrics,
            "threshold": threshold,
            "sorting": sorting,
            "heatmap_cols": heatmap_cols,
            "sample_size": sample_size,
            "precision": precision,
            "show_plot": show_plot,
        }
        not_hash_values = ["metrics", "ngram_range"]

        collect_data_performance(
            scope="sequences",
            event_name="metadata",
            called_params=params,
            not_hash_values=not_hash_values,
            performance_data={},
            eventstream_index=self._eventstream_index,
        )

        self.__sequences = Sequences(eventstream=self)

        self.__sequences.fit(ngram_range=ngram_range, groups=groups, group_names=group_names, path_id_col=weight_col)

        styler_sequences = self.__sequences.plot(
            metrics=metrics,
            threshold=threshold,
            sorting=sorting,
            heatmap_cols=heatmap_cols,
            sample_size=sample_size,
            precision=precision,
        )
        if show_plot:
            display(styler_sequences)
        return self.__sequences

    def segment_map(
        self,
        name: str | None,
        index: Literal["path_id", "segment_value"] = "path_id",
        resolve_collision: Optional[Literal["majority", "last"]] = None,
    ) -> pd.DataFrame | pd.Series:
        """
        Return a mapping between segment values and paths. Works with static or roughly-static segments.

        Parameters
        ----------

        name : str, optional
            A name of the segment. If ``None`` mapping is returned for all segments; works only for ``index="path_id"``.

        index : {"path_id", "segment_value"}, default "path_id".
            The index of the resulting Series or DataFrame. If ``path_id``, the index is path_id,
            and the values are the correspondingsegment values. If ``segment_value``, the index is segment values,
            and the values are lists of path_ids associated with the segment value.

        Returns
        -------

        pd.Series
            If ``name`` is defined.

        pd.DataFrame
            If ``name=None`` and ``index="path_id"``.

        """
        return get_segment_path_map(self, name=name, index=index, resolve_collision=resolve_collision)

    def extract_features(
        self,
        feature_type: FEATURE_TYPES,
        ngram_range: Tuple[int, int] = (1, 1),
        path_id_col: str | None = None,
        col_suffix: str | None = None,
    ) -> pd.DataFrame:
        """
        Calculate set of features for each path.

        Parameters
        ----------
        feature_type : {"tfidf", "count", "frequency", "binary", "markov", "time", "time_fraction"}
            Algorithms for converting event sequences to feature vectors:

            - ``tfidf`` see details in :sklearn_tfidf:`sklearn documentation<>`.
            - ``count`` see details in :sklearn_countvec:`sklearn documentation<>`.
            - ``frequency`` is similar to count, but normalized to the total number of the events
              in the user's trajectory.
            - ``binary`` 1 if a user had the given n-gram at least once and 0 otherwise.
            - ``markov`` available for bigrams only. For a given bigram ``(A, B)`` the vectorized values
              are the user's transition probabilities from ``A`` to ``B``.
            - ``time`` associated with unigrams only. The total amount of time (in seconds) spent
              on a given event.
            - ``time_fraction`` the same as ``time`` but divided by the path duration (in seconds).

        ngram_range : Tuple(int, int), default (1, 1)
            The lower and upper boundary of the range of n for n-grams to be extracted.
            For example, ngram_range=(1, 1) means only single events, (1, 2) means single events
            and bigrams. Ignored for ``markov``, ``time``, ``time_fraction`` feature types.

        path_id_col : str, optional
            A column name associated with a path identifier. A default value is linked to the user column
            from eventstream schema.

        col_suffix : str, optional
            A suffix added to the feature names.


        Returns
        -------
        pd.DataFrame
            A DataFrame with the vectorized values. The index consists of path ids, the columns relate to the n-grams.
        """

        pf = Vectorizer(self)
        features = pf.extract_features(feature_type, ngram_range, path_id_col, col_suffix)
        return features

    def clusters_overview(
        self,
        segment_name: str,
        features: pd.DataFrame,
        aggfunc: Callable | str = "mean",
        scaler: Literal["minmax", "std"] = "minmax",
        metrics: List[Tuple[Union[str, Callable, pd.NamedAgg], Union[str, Callable], str]] | None = None,
        axis: int = 1,
        show_plot: bool = True,
    ) -> SegmentOverview:
        """
        Show a heatmap table with aggregated values of features and custom metrics by clusters.

        Parameters
        ----------
        segment_name : str
            A name of the segment containing cluster labels.

        features : pd.DataFrame
            A DataFrame with features to be aggregated. The DataFrame's index should be path ids.

        aggfunc : callable or str, default "mean"
            A function to aggregate the features. If a string is passed, it should be a valid function name
            for the DataFrame's ``agg`` method, see :pandas_aggfunc:`pandas documentation<>`.
            for the details.

        scaler : {"minmax", "std"}, default "minmax"
            A scaler to normalize the features before the aggregation. Available scalers:

            - ``minmax``: MinMaxScaler.
            - ``std``: StandardScaler.

        metrics : list of tuples, optional
            A list of tuples with custom metrics. Each tuple should contain three elements:

            - a function to calculate the metric, see :py:meth:`Eventstream.path_metrics()<retentioneering.eventstream.eventstream.Eventstream.path_metrics>` for the details;
            - an aggregation metric to be applied to the metric values, same as ``aggfunc``;
            - a metric label to be displayed in the resulting table.

        axis : {0, 1}, default 1
            The axis for which the heatmap is to be generated.

            - 1 : for row-wise heatmap,
            - 0 : for column-wise heatmap. Custom metrics coloring is ignored in this case.

        show_plot : bool, default True
            If ``True``, a heatmap is shown.

        Returns
        -------
        SegmentOverview
            A ``SegmentOverview`` class instance fitted with given parameters.
        """
        self.__segment_overview = SegmentOverview(eventstream=self)
        self.__segment_overview.fit_clusters_overview(segment_name, features, aggfunc, scaler, metrics, axis)
        if show_plot:
            fig = self.__segment_overview.plot_heatmap()
            fig.show()
        return self.__segment_overview

    def segment_overview(
        self,
        segment_name: str,
        metrics: List[Tuple[Union[str, Callable, pd.NamedAgg], Union[str, Callable], str]] | None = None,
        kind: SEGMENT_METRICS_OVERVIEW_KINDS = "heatmap",
        axis: int = 0,
        show_plot: bool = True,
    ) -> SegmentOverview:
        """
        Show a visualization with aggregated values of custom metrics by segments.

        segment_name : str
            A name of the segment.

        metrics : list of tuples, optional
            A list of tuples with custom metrics. Each tuple should contain three elements:

            - a function to calculate the metric, see :py:meth:`Eventstream.path_metrics()<retentioneering.eventstream.eventstream.Eventstream.path_metrics>` for the details;
            - an aggregation metric to be applied to the metric values;
            - a metric label to be displayed in the resulting table.

        kind : {"heatmap", "bar"}, default="heatmap"
            Visualization option.

        axis : {0, 1}, default 0
            The axis for which the heatmap is to be generated.

            - 0 : for row-wise heatmap.
            - 1 : for column-wise heatmap,

        Returns
        -------
        SegmentOverview
            A ``SegmentOverview`` class instance fitted with given parameters.
        """
        self.__segment_overview = SegmentOverview(eventstream=self)

        cm = self.__segment_overview.fit_segment_overview(segment_name, metrics)
        if show_plot:
            fig = self.__segment_overview.plot_custom_metrics(axis, kind)
            fig.show()
        return self.__segment_overview

    def segment_diff(
        self,
        segment_items: SplitExpr,
        features: pd.DataFrame,
        aggfunc: Callable = np.mean,
        threshold: float = 0.01,
        top_n: int | None = None,
        show_plot: bool = True,
    ) -> SegmentDiff:
        """
        Show a table with the difference between a pair of segment items. The rows relate to the features.
        Wasserstein distance is used to calculate the difference between the feature distributions
        of the pair segment items.

        segment_items : list
            A list with segment values to be compared.

        features : pd.DataFrame
            A DataFrame with features to be aggregated and compared between selected segment items.

        aggfunc : callable, default np.mean
            A function to aggregate the features. If a string is passed, it should be a valid function name
            for the DataFrame's ``agg`` method, see :pandas_aggfunc:`pandas documentation<>`.
            for the details.

        threshold : float, default 0.01
            A threshold to filter out the features with a small difference between the segment items.

        top_n : int, optional
            A number of top features to be displayed.

        show_plot : bool, default True
            If ``True``, a table with the difference is shown.

        Returns
        -------
        SegmentDiff
            A ``SegmentDiff`` class instance fitted with given parameters.
        """
        self.__segment_diff = SegmentDiff(self)
        diff = self.__segment_diff.fit(features, segment_items, aggfunc, threshold, top_n)
        if show_plot:
            display(HTML(diff.to_html(escape=False)))
        return self.__segment_diff

    def projection(
        self,
        features: pd.DataFrame,
        method: Literal["tsne", "umap"] = "tsne",
        segments: List[str] | None = None,
        sample_size: int | None = None,
        random_state: int | None = None,
        show_plot: bool = True,
        **kwargs: Any,
    ) -> SegmentProjection:
        """
        Project a dataset to a 2D space using a manifold transformation.

        Parameters
        ----------

        features: pandas.DataFrame
            A dataset to be projected. The index should be path ids.

        method : {"umap", "tsne"}, default "tsne"
            Type of manifold transformation. See :sklearn_tsne:`sklearn.manifold.TSNE()<>` and :umap:`umap.UMAP()<>`
            for the details.

        sample_size : int, optional, default=1000
            The number of elements to sample.

        random_state : int, optional
            Use an int number to make the randomness deterministic. Calling the method multiple times with the same
            ``random_state`` yields the same results.

        **kwargs : optional
            Additional parameters for :sklearn_tsne:`sklearn.manifold.TSNE()<>` and :umap:`umap.UMAP()<>`.

        Returns
        -------
        SegmentProjection
            A ``SegmentProjection`` class instance fitted with given parameters.
        """
        pf = Vectorizer(self)
        projection_2d = pf.projection(features, method, sample_size, random_state, **kwargs)

        self.__segment_projection = SegmentProjection(self)
        self.__segment_projection.fit(projection_2d, segments)

        if show_plot:
            fig = self.__segment_projection.plot()
            fig.show()

        return self.__segment_projection

    def path_metrics(
        self,
        metrics: Tuple[PATH_METRIC_TYPES, str] | List,
        path_id_col: str | None = None,
    ) -> pd.DataFrame | pd.Series:
        """
        Calculate metrics for each path.

        Parameters
        ----------
        metrics : tuple, or list
            A metric or a list of metrics to be calculated.

            Each metric can be defined with the following types.

            - str. The following metric aliases are supported.
                - ``len``: the number of events in the path.
                - ``has:TARGET_EVENT``: whether the path contains the specified target event.
                - ``time_to:TARGET_EVENT``: the time to the first occurrence of the specified target event.
            - pd.NamedAgg. It is applied to a single column of the grouped DataFrame. See :pandas_namedagg:`pandas documentation<>` for the details.
            - Callable. An arbitrary function to be applied to the grouped DataFrame with apply method.

            A metric should be passed as a tuple of two elements:
                - a metric definition, according to the mentioned types.
                - a metric name.

            Examples of the metrics:

            .. code:: python

                metrics = [
                    ('len', 'path_length'),
                    ('has:cart', 'has_cart'),
                    ('time_to:cart', 'time_to_cart'),
                    (lambda _df: (_df['event'] == 'cart').sum(), 'cart_count'),
                    (pd.NamedAgg('timestamp', lambda s: len(s.dt.date.unique())), 'active_days')
                ]

        path_id_col : str, optional
            A column name associated with a path identifier. A default value is linked to the user column
            from eventstream schema.

        Returns
        -------
        pd.DataFrame or pd.Series
            A DataFrame (for multiple metrics) or Series (for a single metric) with the calculated metric values. The index consists of path ids.
        """
        df = self.to_dataframe()
        event_col = self.schema.event_name
        time_col = self.schema.event_timestamp

        if path_id_col is None:
            path_id_col = self.schema.user_id

        metric_list = metrics if isinstance(metrics, list) else [metrics]

        res = pd.DataFrame()
        for metric_item in metric_list:
            if isinstance(metric_item, Tuple):  # type: ignore
                metric, metric_name = metric_item
            else:
                raise TypeError(f"Metric is supposed to be a str or tuple. Got {type(metric_item)} instead.")

            if isinstance(metric, str):  # type: ignore
                if metric == "len":
                    metric_values = df.groupby(path_id_col).size().rename("len")
                elif metric.startswith(METRIC_PREFIX_HAS):
                    target_event = metric.split(METRIC_PREFIX_HAS)[1]
                    metric_values = (
                        df.assign(is_target_event=lambda _df: _df[event_col] == target_event)
                        .groupby(path_id_col)["is_target_event"]
                        .any()
                    )
                elif metric.startswith(METRIC_PREFIX_TIME_TO_EVENT):
                    target_event = metric.split(METRIC_PREFIX_TIME_TO_EVENT)[1]

                    df = df.assign(is_target_event=lambda _df: _df[event_col] == target_event)
                    first_timestamp = df.groupby(path_id_col)[time_col].min().rename("first_timestamp")
                    target_timestamp = (
                        df[df[event_col] == target_event]
                        .groupby(path_id_col)[time_col]
                        .min()
                        .rename("target_timestamp")
                    )
                    timestamps = first_timestamp.to_frame("first_timestamp").join(
                        target_timestamp.to_frame("target_timestamp"), how="left"
                    )
                    timestamps[metric_name] = timestamps["target_timestamp"] - timestamps["first_timestamp"]
                    metric_values = timestamps[metric_name]
                else:
                    raise TypeError(f"Metric alias '{metric}' is not supported.")
            elif isinstance(metric, LambdaType) or isinstance(metric, FunctionType):
                metric_values = df.groupby(path_id_col).apply(metric, include_groups=False)
            elif isinstance(metric, pd.NamedAgg):
                metric_values = df.groupby(path_id_col).agg(**{metric_name: metric})  # type: ignore
            else:
                raise TypeError(
                    f"A metric is supposed to be one of the following types: {get_args(PATH_METRIC_TYPES)}."
                )
            metric_values.name = metric_name
            res = pd.concat([res, metric_values], axis=1)

        # return Series if a single metric was passed
        if not isinstance(metrics, list):
            return res[res.columns[0]]

        return res
