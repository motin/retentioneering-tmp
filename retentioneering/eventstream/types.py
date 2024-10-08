from __future__ import annotations

import collections
import collections.abc
import typing
from abc import abstractmethod
from dataclasses import field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    Tuple,
    TypedDict,
    Union,
)

import pandas as pd

from retentioneering.common.constants import FEATURE_TYPES, PATH_METRIC_TYPES

IndexOrder = List[Optional[str]]
OrderDict = Dict[str, int]


class EventstreamType(Protocol):
    schema: EventstreamSchemaType
    index_order: OrderDict
    __raw_data_schema: RawDataSchemaType
    __events: pd.DataFrame | pd.Series[Any]

    @abstractmethod
    def copy(self) -> EventstreamType:
        ...

    @property
    @abstractmethod
    def _eventstream_index(self) -> int:
        ...

    @abstractmethod
    def append_eventstream(self, eventstream: EventstreamType) -> None:
        ...

    @property
    @abstractmethod
    def _hash(self) -> str:
        ...

    @abstractmethod
    def to_dataframe(self, copy: bool = False, drop_segment_events: bool = True) -> pd.DataFrame:
        ...

    @abstractmethod
    def add_custom_col(self, name: str, data: pd.Series[Any] | None) -> None:
        ...

    @abstractmethod
    def segment_map(
        stream: EventstreamType,
        label: str,
        index: Literal["path_id", "segment_value"],
        resolve_collision: Optional[Literal["majority", "last"]] = None,
    ) -> pd.DataFrame | pd.Series:
        ...

    @abstractmethod
    def extract_features(
        self,
        feature_type: FEATURE_TYPES,
        ngram_range: Tuple[int, int],
        path_id_col: str | None = None,
        col_suffix: str | None = None,
    ) -> pd.DataFrame:
        ...

    @abstractmethod
    def path_metrics(
        self,
        metrics: Tuple[PATH_METRIC_TYPES, str] | List,
        path_id_col: str | None = None,
    ) -> pd.DataFrame | pd.Series:
        ...

    @abstractmethod
    def add_segment(self, segment: AddSegmentType, name: Optional[str] = None) -> EventstreamType:
        ...


class EventstreamSchemaType(Protocol):
    custom_cols: List[str] = field(default_factory=list)
    user_id: str = "user_id"
    event_timestamp: str = "event_timestamp"
    event_name: str = "event_name"
    event_index: str = "event_index"
    event_type: str = "event_type"
    event_id: str = "event_id"

    @abstractmethod
    def copy(self) -> EventstreamSchemaType:
        ...

    @abstractmethod
    def is_equal(self, schema: EventstreamSchemaType) -> bool:
        ...

    @abstractmethod
    def get_default_cols(self) -> List[str]:
        ...

    @abstractmethod
    def get_cols(self) -> list[str]:
        ...

    @abstractmethod
    def to_raw_data_schema(self, event_id: bool = False, event_index: bool = False) -> RawDataSchemaType:
        ...


class RawDataCustomColSchema(TypedDict):
    raw_data_col: str
    custom_col: str


class RawDataSchemaType(Protocol):
    event_name: str = "event"
    event_timestamp: str = "timestamp"
    user_id: str = "user_id"
    event_index: Optional[str] = None
    event_type: Optional[str] = None
    event_id: Optional[str] = None
    custom_cols: List[RawDataCustomColSchema] = field(default_factory=list)

    @abstractmethod
    def get_default_cols(self) -> List[str]:
        ...

    @abstractmethod
    def copy(self) -> RawDataSchemaType:
        ...


class SeriesWithValidator(pd.Series):  # pyright: ignore [reportUntypedBaseClass]
    @classmethod
    def __get_validators__(
        cls: typing.Type["SeriesWithValidator"],
    ) -> collections.abc.Iterable[collections.abc.Callable]:
        yield cls.validate_custom_class

    @classmethod
    def validate_custom_class(cls: typing.Type["SeriesWithValidator"], passed_value: typing.Any) -> pd.Series:
        if isinstance(passed_value, pd.Series):
            return passed_value

        raise ValueError


AddSegmentType = Union[str, Callable, SeriesWithValidator]
SplitExpr = Tuple[str, Any, Any]
SegmentFilterExpr = Tuple[str, Union[str, list]]
UserListType = List[Union[str, int]]
UserGroupsType = List[UserListType]
UserGroupsNamesType = Union[List[str], Tuple[str, str]]
