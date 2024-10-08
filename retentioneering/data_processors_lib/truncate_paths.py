from __future__ import annotations

from typing import Any, Literal, Optional

import pandas as pd

from retentioneering.backend.tracker import collect_data_performance, time_performance
from retentioneering.data_processor import DataProcessor
from retentioneering.eventstream.segments import (
    _calculate_segment_col,
    _get_segments_last_values,
)
from retentioneering.eventstream.types import EventstreamSchemaType
from retentioneering.params_model import ParamsModel
from retentioneering.utils.doc_substitution import docstrings
from retentioneering.utils.hash_object import hash_dataframe


class TruncatePathsParams(ParamsModel):
    """
    A class with parameters for :py:class:`.TruncatePath` class.
    """

    drop_before: Optional[str]
    drop_after: Optional[str]
    occurrence_before: Literal["first", "last"] = "first"
    occurrence_after: Literal["first", "last"] = "first"
    shift_before: int = 0
    shift_after: int = 0
    ignore_before: bool = False
    ignore_after: bool = False
    keep_synthetic: bool = False


@docstrings.get_sections(base="TruncatePath")  # type: ignore
class TruncatePaths(DataProcessor):
    """
    Leave a sub-path bounded with a given conditions. Left and right boundary conditions are associated
    with ``before`` and ``after`` argument suffixes correspondingly. If any of ``before`` and ``after``
    conditions do not meet, the path is excluded from the resulting eventstream entirely.

    Parameters
    ----------
    drop_before : str, optional
        Event name before which part of the user's path is dropped. The specified event remains in the data.
    drop_after : str, optional
        Event name after which part of the user's path is dropped. The specified event remains in the data.
    occurrence_before : {"first", "last"}, default "first"
        This parameter is necessary when the specified event occurs more than once in one user's path.

        - when set to ``first``, the part of the user path before the first event occurrence is dropped;
        - when set to ``last``, the part of the user path before the last event occurrence is dropped;
    occurrence_after : {"first", "last"}, default "first"
        The same behavior as in the 'occurrence_before', but for the other part of the user path.
    shift_before : int,  default 0
        Sets the number of steps by which the truncate point is shifted from the selected event.
        If the value is negative, then the offset occurs to the left along the timeline.
        If positive, then it occurs to the right.
    shift_after : int,  default 0
        The same behavior as in the ``shift_before``, but for the other part of the user path.
    ignore_before : bool,  default False
        If True, the resulting evenstream includes the paths that do not meet ``before`` condition.
    ignore_after : bool,  default False
        If True, the resulting evenstream includes the paths that do not meet ``after`` condition.
    keep_synthetic : bool, default False
        If True, all the synthetic events that are associated with the boundary events are kept in the output
        eventstream.

    Returns
    -------
    Eventstream
        ``Eventstream`` with the truncated sub-paths.

    Notes
    -----
    - See :doc:`Data processors user guide</user_guides/dataprocessors>` for the details.
    - If a path contains a segment synthetic event before a ``before`` cut point, the last segment value
      that occurred before the cut point is inherited. See :doc:`segments user guide</user_guides/segments_and_clusters>`.

    """

    params: TruncatePathsParams

    @time_performance(
        scope="truncate_paths",
        event_name="init",
    )
    def __init__(self, params: TruncatePathsParams):
        super().__init__(params=params)

    @time_performance(
        scope="truncate_paths",
        event_name="apply",
    )
    def apply(self, df: pd.DataFrame, schema: EventstreamSchemaType) -> pd.DataFrame:
        user_col = schema.user_id
        event_col = schema.event_name
        event_type_col = schema.event_type

        drop_before = self.params.drop_before
        drop_after = self.params.drop_after
        occurrence_before = self.params.occurrence_before
        occurrence_after = self.params.occurrence_after
        shift_before = self.params.shift_before
        shift_after = self.params.shift_after
        ignore_before = self.params.ignore_before
        ignore_after = self.params.ignore_after
        keep_synthetic = self.params.keep_synthetic

        params_data: list[Any] = []

        if not drop_after and not drop_before:
            raise Exception("Either drop_before or drop_after must be specified!")

        if not keep_synthetic:
            df["new_index"] = 1
            df["new_index"] = df.groupby(user_col)["new_index"].cumsum()
            index_col = "new_index"
        else:
            index_col = schema.event_index

        # tm = truncate marks
        tm = df.copy()

        if drop_before:
            before: list[str | list[str | int | None]] | None = [
                drop_before,
                ["before", occurrence_before, shift_before],
            ]
            params_data.append(before)

        if drop_after:
            after: list[str | list[str | int | None]] | None = [drop_after, ["after", occurrence_after, shift_after]]
            params_data.append(after)

        for truncate_type in params_data:
            col_mark, occurrence, shift = truncate_type[1]

            if truncate_type[0]:
                mask_events = tm[event_col] == truncate_type[0]
                tm[f"{col_mark}_mark_target"] = mask_events.astype(int)
                tm[f"{col_mark}_mark_target"] = tm.groupby([user_col, index_col])[f"{col_mark}_mark_target"].transform(
                    max
                )
                if occurrence == "last":
                    tm[f"{col_mark}_cumsum"] = tm.iloc[::-1].groupby([user_col])[f"{col_mark}_mark_target"].cumsum()
                if occurrence == "first":
                    tm[f"{col_mark}_cumsum"] = tm.groupby([user_col])[f"{col_mark}_mark_target"].cumsum()

                def count_groups(x: pd.DataFrame) -> int:
                    return x.to_frame(name=index_col).groupby(index_col).ngroup()  # type: ignore

                tm[f"{col_mark}_group_num_in_user"] = tm.groupby([user_col], group_keys=False)[index_col].transform(
                    count_groups
                )

                if occurrence == "last":
                    df_groups = (
                        tm[tm[f"{col_mark}_cumsum"] == 1]
                        .groupby([user_col])[f"{col_mark}_group_num_in_user"]
                        .max()
                        .rename(f"{col_mark}_group_centered")
                        .reset_index()
                    )
                else:
                    df_groups = (
                        tm[tm[f"{col_mark}_cumsum"] == 1]
                        .groupby([user_col])[f"{col_mark}_group_num_in_user"]
                        .min()
                        .rename(f"{col_mark}_group_centered")
                        .reset_index()
                    )

                tm = tm.merge(df_groups, how="left")
                tm[f"{col_mark}_group_centered"] = (
                    tm[f"{col_mark}_group_num_in_user"] - tm[f"{col_mark}_group_centered"] - shift
                )

        mask = pd.Series([True] * len(tm))
        if drop_before:
            if not ignore_before:
                mask &= tm["before_group_centered"] >= 0
            else:
                mask &= (tm["before_group_centered"] >= 0) | (tm["before_group_centered"].isna())

            last_segments = _get_segments_last_values(tm[tm["before_cumsum"] == 0], schema=schema)
            mask = mask | tm.index.isin(last_segments.index)

        if drop_after:
            if not ignore_after:
                mask &= tm["after_group_centered"] <= 0
            else:
                mask &= (tm["after_group_centered"] <= 0) | (tm["after_group_centered"].isna())

        if not keep_synthetic:
            df = df.drop("new_index", axis=1)

        result = df[mask]

        collect_data_performance(
            scope="truncate_paths",
            event_name="metadata",
            called_params=self.to_dict()["values"],
            not_hash_values=["occurrence_before", "occurrence_after"],
            performance_data={
                "parent": {
                    "shape": df.shape,
                    "hash": hash_dataframe(df),
                },
                "child": {
                    "shape": result.shape,
                    "hash": hash_dataframe(result),
                },
            },
        )

        return result
