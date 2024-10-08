from __future__ import annotations

from typing import Any, Literal, Optional, Tuple

import pandas as pd
from typing_extensions import TypeGuard

from retentioneering.eventstream.types import (
    EventstreamSchemaType,
    EventstreamType,
    SegmentFilterExpr,
    SplitExpr,
    UserGroupsNamesType,
    UserGroupsType,
)

SEGMENT_TYPE = "segment"
SEGMENT_DELIMITER = "::"
SEGM_NAME_COL = "segment_name"
SEGM_VALUE_COL = "segment_value"

OUTER_LITERAL = "_OUTER_"
ALL_LITERAL = "_ALL_"


def _create_segment_event(event_col: pd.Series, name: str) -> pd.Series:
    return name + SEGMENT_DELIMITER + event_col.astype(str)


def _get_segment_mask(
    df: pd.DataFrame, schema: EventstreamSchemaType, name: Optional[str] = None, check_existence: bool = True
) -> pd.Series[bool]:
    if not name:  # return all segments
        mask = df[schema.event_type] == SEGMENT_TYPE
    else:
        mask = (df[schema.event_type] == SEGMENT_TYPE) & (_extract_segment_keys(df.loc[:, schema.event_name]) == name)

    if check_existence and (~mask).all():  # type: ignore
        raise KeyError(f"segment with name {name} is non-existent!")

    return mask


def _extract_segment_values(col: pd.Series) -> pd.Series:
    return col.astype(str).str.split(SEGMENT_DELIMITER).str[-1]


def _extract_segment_keys(col: pd.Series) -> pd.Series:
    return col.astype(str).str.split(SEGMENT_DELIMITER).str[0]


def _get_segments_last_values(df: pd.DataFrame, schema: EventstreamSchemaType) -> pd.DataFrame:
    event_type_col = schema.event_type
    user_col = schema.user_id
    event_col = schema.event_name

    new_df = df[df[event_type_col] == SEGMENT_TYPE][[user_col, event_col]].copy()
    new_df["segment_name"] = _extract_segment_keys(new_df.loc[:, event_col])

    return new_df.groupby([user_col, "segment_name"]).tail(1)


def _calculate_segment_col(df: pd.DataFrame, schema: EventstreamSchemaType, name: str) -> pd.Series[Any]:
    segm_mask = _get_segment_mask(df, schema, name)  # todo @dakhaytin - refactor this check
    tmp = df[[schema.user_id, schema.event_index]].copy()

    tmp["is_current_segment"] = _extract_segment_keys(df.loc[:, schema.event_name]) == name
    tmp["segment_cumsum"] = tmp.groupby(schema.user_id)["is_current_segment"].cumsum()
    tmp["segment_cumsum"] = tmp[schema.user_id].astype(str) + tmp["event_index"].map(
        tmp.groupby("event_index")["segment_cumsum"].max()
    ).astype(
        str
    )  # fill gaps for segment events before target segment

    mask = tmp["is_current_segment"]
    tmp.loc[mask, "segment_value"] = _extract_segment_values(df.loc[mask][schema.event_name])
    tmp[name] = tmp["segment_cumsum"].map(tmp.groupby("segment_cumsum")["segment_value"].first())

    return tmp[name]


def _filter_segment_events(df: pd.DataFrame, schema: EventstreamSchemaType, segment: SegmentFilterExpr) -> None:
    name, values = segment
    if isinstance(values, str):
        values = [values]

    segment_col = _calculate_segment_col(df, schema, name)

    mask = segment_col.isin(values)
    df.drop(df.loc[~mask].index, inplace=True)


def _extract_groups(path_df: pd.Series, value_l: str, value_r: str) -> UserGroupsType:
    if value_l in [OUTER_LITERAL, ALL_LITERAL]:
        raise KeyError("Left value have to be exact value of chosen segment!")

    group_l = path_df.loc[path_df.index == value_l].unique().tolist()

    if value_r == OUTER_LITERAL:
        group_r = path_df.loc[path_df.index != value_l].unique().tolist()
    elif value_r == ALL_LITERAL:
        group_r = path_df.loc[:].unique().tolist()
    else:
        group_r = path_df.loc[path_df.index == value_r].unique().tolist()

    return group_l, group_r  # type: ignore


def _split_segment(
    stream: EventstreamType, split_expr: SplitExpr | UserGroupsType | str
) -> Tuple[UserGroupsType, UserGroupsNamesType]:
    df: pd.Series

    if _is_user_groups(split_expr):
        return split_expr, []

    elif isinstance(split_expr, str):  # if binary segment
        df = get_segment_path_map(stream, split_expr, "segment_value")  # type: ignore
        if df.index.nunique() != 2:
            raise ValueError(f"{split_expr} is not a binary segment!")

        value_l, value_r = df.index.unique().tolist()

    elif _is_split_expr(split_expr):  # if split expression
        name, value_l, value_r = split_expr
        df = get_segment_path_map(stream, name, "segment_value")  # type: ignore

    else:
        raise ValueError("Wrong user groups input!")

    groups = _extract_groups(df, value_l, value_r)

    if len(groups[0]) == 0:
        raise ValueError(f"Segment with value {value_l} is empty!")
    elif len(groups[1]) == 0:
        raise ValueError(f"Segment with value {value_r} is empty!")

    return groups, [value_l, value_r]  # type: ignore


def _is_user_groups(split_exp: SplitExpr | UserGroupsType | str) -> TypeGuard[UserGroupsType]:
    check = all(isinstance(expr, list) for expr in split_exp) and len(split_exp) == 2
    return check


def _is_split_expr(split_exp: SplitExpr | UserGroupsType | str) -> TypeGuard[SplitExpr]:
    check = (
        len(split_exp) == 3
        and isinstance(split_exp[0], str)
        and all(isinstance(expr, (str, int, float)) for expr in split_exp[1:])
    )
    return check


# PUBLIC API STARTS HERE #
def get_segment_path_map(
    stream: EventstreamType,
    name: str | None = None,
    index: Literal["path_id", "segment_value"] = "path_id",
    resolve_collision: Optional[Literal["majority", "last"]] = None,
) -> pd.DataFrame | pd.Series:
    df = stream.to_dataframe(drop_segment_events=False)
    schema = stream.schema
    user_col = stream.schema.user_id
    event_col = stream.schema.event_name

    df = df.loc[_get_segment_mask(df, schema, name)][[schema.user_id, schema.event_name]].drop_duplicates().copy()
    df[SEGM_NAME_COL] = _extract_segment_keys(df.loc[:, schema.event_name])
    df[schema.event_name] = _extract_segment_values(df.loc[:, schema.event_name])

    if not resolve_collision:
        ...

    elif resolve_collision == "majority":
        df = df.groupby([user_col, SEGM_NAME_COL])[event_col].agg(lambda x: x.value_counts().index[0]).reset_index()

    elif resolve_collision == "last":
        df = df.groupby([user_col, SEGM_NAME_COL])[event_col].last().reset_index()

    else:
        raise KeyError('resolve_collision option should be either "majority" or "last"!')

    df = df.rename(columns={event_col: SEGM_VALUE_COL})

    if not name:
        df = df.sort_values(user_col).reset_index(drop=True)
        df = df[[user_col, SEGM_NAME_COL, SEGM_VALUE_COL]]
        return df

    if index == "path_id":
        df.set_index(user_col, inplace=True)
        return_col = SEGM_VALUE_COL
    elif index == "segment_value":
        df.set_index(SEGM_VALUE_COL, inplace=True)
        return_col = user_col
    else:
        raise KeyError("index must be either path_id or segment_value")

    df.sort_index(inplace=True)
    return df[return_col]


def materialize_segment(stream: EventstreamType, name: str) -> None:
    df = stream.to_dataframe(drop_segment_events=False)
    schema = stream.schema

    segment_col = _calculate_segment_col(df, schema, name)
    stream.add_custom_col(name, segment_col)


def get_all_segments(stream: EventstreamType, resolve_collision: Literal["majority", "last"]) -> pd.DataFrame:
    user_col = stream.schema.user_id
    event_col = stream.schema.event_name

    all_segments = get_segment_path_map(stream, None, "path_id", resolve_collision)

    all_segments = all_segments.pivot(index=user_col, columns=SEGM_NAME_COL, values=SEGM_VALUE_COL)

    return all_segments
