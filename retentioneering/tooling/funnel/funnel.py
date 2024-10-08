from __future__ import annotations

from collections.abc import Collection
from copy import deepcopy
from typing import Any, Literal

import pandas as pd
import plotly.graph_objects as go
from pandas.core.common import flatten

from retentioneering.backend.tracker import (
    collect_data_performance,
    time_performance,
    track,
)
from retentioneering.eventstream.segments import _split_segment
from retentioneering.eventstream.types import (
    EventstreamType,
    SplitExpr,
    UserGroupsNamesType,
    UserGroupsType,
    UserListType,
)

FunnelTypes = Literal["open", "closed", "hybrid"]


class Funnel:
    """
    A class for the calculation and visualization of a conversion funnel.

    Parameters
    ----------
    eventstream : EventstreamType

    See Also
    --------
    .Eventstream.funnel : Call Funnel tool as an eventstream method.

    Notes
    -----
    See :doc:`Funnel user guide</user_guides/funnel>` for the details.

    """

    __default_layout = dict(
        margin={"l": 180, "r": 0, "t": 30, "b": 0, "pad": 0},
        funnelmode="stack",
        showlegend=True,
        hovermode="closest",
        legend=dict(orientation="v", bgcolor="#E2E2E2", xanchor="left", font=dict(size=12)),
    )
    __eventstream: EventstreamType
    stages: list[str]
    stage_names: list[str] | None
    funnel_type: FunnelTypes
    groups: UserGroupsType
    group_names: UserGroupsNamesType | None
    __res_dict: dict[str, dict]

    @time_performance(
        scope="funnel",
        event_name="init",
    )
    def __init__(self, eventstream: EventstreamType) -> None:
        self.__eventstream = eventstream
        self.user_col = self.__eventstream.schema.user_id
        self.event_col = self.__eventstream.schema.event_name
        self.time_col = self.__eventstream.schema.event_timestamp
        self.__res_dict = {}

    def __validate_input(
        self,
        stages: list[str],
        stage_names: list[str] | None = None,
        funnel_type: FunnelTypes = "closed",
        groups_input: SplitExpr | UserGroupsType | None = None,
        group_names: UserGroupsNamesType | None = None,
    ) -> tuple[pd.DataFrame, list[str], list[str], FunnelTypes, UserGroupsType, UserGroupsNamesType]:
        data = self.__eventstream.to_dataframe(copy=True)
        data = data[data[self.event_col].isin([i for i in flatten(stages)])]  # type: ignore

        if stages and stage_names and len(stages) != len(stage_names):
            raise ValueError("stages and stage_names must be the same length!")

        if groups_input is None:
            groups = [data[self.user_col].unique().tolist()]
            group_names = ["all_users"]
        else:
            groups, group_names_ = _split_segment(self.__eventstream, groups_input)
            group_names = group_names_ if group_names_ else group_names

        if group_names is None:
            group_names = [f"group_{i}" for i in range(len(groups))]  # type: ignore

        if groups and group_names and len(groups) != len(group_names):  # type: ignore
            raise ValueError("groups_description and group_names must be the same length!")

        # IDK why but pyright thinks this is Funnel!!!

        if funnel_type not in ["open", "closed", "hybrid"]:
            raise ValueError("funnel_type should be 'open', 'closed' or 'hybrid'!")

        stages_2d_list = deepcopy(stages)
        for idx, stage in enumerate(stages_2d_list):
            if type(stage) is not list:
                stages_2d_list[idx] = [stage]  # type: ignore

        if stage_names is None:
            stage_names = []
            for t in stages_2d_list:
                # get name
                stage_names.append(" | ".join(t).strip(" | "))

        return data, stages_2d_list, stage_names, funnel_type, groups, group_names

    def _plot_stacked_funnel(self, data: list[go.Funnel]) -> go.Figure:
        layout = go.Layout(**self.__default_layout)
        fig = go.Figure(data, layout)
        return fig

    @staticmethod
    def _calculate_plot_data(plot_params: dict[str, Any]) -> list[go.Funnel]:
        data = []
        for t in plot_params.keys():
            trace = go.Funnel(
                name=t,
                y=plot_params[t]["stages"],
                x=plot_params[t]["values"],
                textinfo="value+percent initial+percent previous",
            )
            data.append(trace)

        return data

    def _prepare_data_for_closed_and_hybrid_funnel(
        self,
        data: pd.DataFrame,
        stages: list[str],
        stage_names: list[str],
        groups: UserGroupsType,
        group_names: UserGroupsNamesType,
    ) -> dict[str, dict]:
        min_time_0stage = (
            data[data[self.event_col].isin(stages[0])].groupby(self.user_col)[[self.time_col]].min().reset_index()
        )
        data = data.merge(min_time_0stage, "left", on=self.user_col, suffixes=("", "_min"))
        data.rename(columns={data.columns[-1]: "min_date"}, inplace=True)

        # filtered NA and only events that occurred after the user entered the first funnel event remain
        data = data[(~data["min_date"].isna()) & (data["min_date"] <= data[self.time_col])]
        data.drop(columns="min_date", inplace=True)

        __res_dict = {}
        for segment, name in zip(groups, group_names):
            vals, _df = self._crop_df(data, stages, segment)
            __res_dict[name] = {"stages": stage_names, "values": vals}
        return __res_dict

    def _prepare_data_for_open_funnel(
        self,
        data: pd.DataFrame,
        stages: list[str],
        stage_names: list[str],
        groups: UserGroupsType,
        group_names: UserGroupsNamesType,
    ) -> dict[str, dict]:
        __res_dict = {}
        for segment, name in zip(groups, group_names):
            # isolate users from group
            group_data = data[data[self.user_col].isin(segment)]
            vals = [group_data[group_data[self.event_col].isin(stage)][self.user_col].nunique() for stage in stages]
            __res_dict[name] = {"stages": stage_names, "values": vals}
        return __res_dict

    def _crop_df(self, df: pd.DataFrame, stages: list[str], segment: UserListType) -> tuple[list[int], pd.DataFrame]:
        first_stage = stages[0]
        next_stages = stages[1:]

        first_stage_users = set(
            (df[(df[self.event_col].isin(first_stage)) & (df[self.user_col].isin(segment))][self.user_col])
        )
        df = df.drop(
            df[(~df[self.user_col].isin(first_stage_users)) | (df[self.event_col].isin(first_stage))].index.tolist()
        )

        prev_users_stage = first_stage_users
        vals = [len(first_stage_users)]
        for stage in next_stages:
            user_stage = set(
                df[(df[self.event_col].isin(stage)) & (df[self.user_col].isin(first_stage_users))][self.user_col]
            )
            user_stage = user_stage - (user_stage - prev_users_stage)
            prev_users_stage = user_stage

            vals.append(len(user_stage))

            if self.funnel_type == "closed":
                stage_min_df = (
                    df[df[self.event_col].isin(stage)].groupby(self.user_col)[[self.time_col]].min().reset_index()
                )
                df = df.merge(stage_min_df, "left", on=self.user_col, suffixes=("", "_min"))
                df.rename(columns={df.columns[-1]: "min_date"}, inplace=True)

                df.drop(
                    df[
                        (df["min_date"].isna())
                        | (df["min_date"] >= df[self.time_col])
                        | (~df[self.user_col].isin(user_stage))
                    ].index.tolist(),
                    inplace=True,
                )
                df.drop(columns="min_date", inplace=True)
            else:
                df = df.drop(df[~df[self.user_col].isin(user_stage)].index.tolist())

        return vals, df

    @time_performance(
        scope="funnel",
        event_name="fit",
    )
    def fit(
        self,
        stages: list[str],
        stage_names: list[str] | None = None,
        funnel_type: FunnelTypes = "closed",
        groups: SplitExpr | None = None,
        group_names: UserGroupsNamesType | None = None,
    ) -> None:
        """
        Calculate the funnel internal values with the defined parameters.
        Applying ``fit`` method is necessary for the following usage
        of any visualization or descriptive ``Funnel`` methods.

        Parameters
        ----------
        stages : list of str
            List of events used as stages for the funnel. Absolute and relative
            number of users who reached specified events at least once will be
            plotted. Multiple events can be grouped together as an individual state
            by combining them as a sub list.
        stage_names : list of str, optional
            List of stage names, this is necessary for stages that include several events.
        funnel_type : 'open', 'closed' or 'hybrid', default 'closed'

            - If ``open``, all users will be counted on each stage;
            - If ``closed``, each stage will include only users, that were present on all previous stages;
            - If ``hybrid``, combination of 2 previous types. The first stage is required
              to go further. And for the second and subsequent stages it is important to have
              all previous stages in their path, but the order of these events is not taken
              into account.

        groups : tuple[list, list], tuple[str, str, str], str, optional
            Specify two groups of paths to plot grouped funnel.

            - If ``tuple[list, list]``, each sub-list should contain valid path ids.
            - If ``tuple[str, str, str]``, the first str should refer to a segment name,
              the others should refer to the corresponding segment values.
            - If ``str``, it should refer to a binary (i.e. containing two segment values only) segment name.
            - If ``None``, all users from the dataset will be plotted. A user can only belong to one segment at a time.

        group_names : list of str, optional
            Names of the groups defined in the ``group`` argument.
        """
        called_params = {
            "stages": stages,
            "stage_names": stage_names,
            "funnel_type": funnel_type,
            "groups": groups,
            "group_names": group_names,
        }
        not_hash_values = ["funnel_type"]
        (
            data,
            self.stages,
            self.stage_names,
            self.funnel_type,
            self.groups,
            self.group_names,
        ) = self.__validate_input(stages, stage_names, funnel_type, groups, group_names)

        if self.funnel_type in ["closed", "hybrid"]:
            self.__res_dict = self._prepare_data_for_closed_and_hybrid_funnel(
                data=data,
                stages=self.stages,
                stage_names=self.stage_names,
                groups=self.groups,
                group_names=self.group_names,
            )

        elif self.funnel_type == "open":
            self.__res_dict = self._prepare_data_for_open_funnel(
                data=data,
                stages=self.stages,
                groups=self.groups,
                group_names=self.group_names,
                stage_names=self.stage_names,
            )
        collect_data_performance(
            scope="funnel",
            event_name="metadata",
            called_params=called_params,
            not_hash_values=not_hash_values,
            performance_data={},
            eventstream_index=self.__eventstream._eventstream_index,
        )

    @time_performance(
        scope="funnel",
        event_name="plot",
    )
    def plot(self) -> go.Figure:
        """
        Create a funnel plot based on the calculated funnel values.
        Should be used after :py:func:`fit`.

        Returns
        -------
        go.Figure

        """
        result_dict = self.__res_dict
        data = self._calculate_plot_data(plot_params=result_dict)
        figure = self._plot_stacked_funnel(data=data)
        return figure

    @property
    @time_performance(
        scope="funnel",
        event_name="values",
    )
    def values(self) -> pd.DataFrame:
        """
        Returns a pd.DataFrame representing the calculated funnel values.
        Should be used after :py:func:`fit`.

        Returns
        -------
        pd.DataFrame

            +------------------+-------------+-----------------+-------------------+--------------------+
            | **segment_name** |  **stages** | **unique_users**|  **%_of_initial** |  **%_of_previous** |
            +------------------+-------------+-----------------+-------------------+--------------------+
            | segment_1        |  stage_1    |            2000 |            100.00 |          100.00    |
            +------------------+-------------+-----------------+-------------------+--------------------+

        """

        result_dict = self.__res_dict
        result_list = []
        for key in result_dict:
            result_ = pd.DataFrame(result_dict[key])
            result_.columns = ["stages", "unique_users"]  # type: ignore
            result_["segment_name"] = key
            result_ = result_[["segment_name", "stages", "unique_users"]]
            result_["shift"] = result_["unique_users"].shift(periods=1, fill_value=result_["unique_users"][0])
            result_["%_of_previous"] = (result_["unique_users"] / result_["shift"] * 100).round(2)
            result_["%_of_initial"] = (result_["unique_users"] / result_["unique_users"][0] * 100).round(2)
            result_.drop(columns="shift", inplace=True)
            result_list.append(result_)

        result_df = pd.concat(result_list).set_index(["segment_name", "stages"])

        return result_df

    @property
    @time_performance(
        scope="funnel",
        event_name="params",
    )
    def params(self) -> dict:
        """
        Returns the parameters used for the last fitting.

        """

        return {
            "stages": self.stages,
            "stage_names": self.stage_names,
            "funnel_type": self.funnel_type,
            "groups": self.groups,
            "group_names": self.group_names,
        }
