from __future__ import annotations

from typing import Any, Literal, Optional, Tuple, Union

import matplotlib.axes
import networkx as nx
import pandas as pd
import seaborn as sns

from retentioneering.edgelist import Edgelist
from retentioneering.eventstream.helpers import FilterEventsHelperMixin
from retentioneering.eventstream.segments import _split_segment
from retentioneering.eventstream.types import (
    EventstreamType,
    SplitExpr,
    UserGroupsNamesType,
    UserGroupsType,
)
from retentioneering.nodelist import Nodelist
from retentioneering.tooling.typing.transition_graph import NormType

MAX_DIM = 60
SHOW_VALUES_DIM = 30
SEQUENCES_URL = "https://doc.retentioneering.com/stable/doc/user_guides/sequences.html"
TRANSITION_MATRIX_VALUES_URL = "https://doc.retentioneering.com/stable/doc/user_guides/transition_matrix.html#values"


class TransitionMatrix:
    """
    The TransitionMatrix class represents a matrix where the element at position (i, j)
    displays the weight of the transition from event i to event j.
    This class provides methods for calculating and visualizing transition matrices,
    using the same logic as for calculating edge weights in a transition graph.

    Parameters
    ----------
    eventstream : EventstreamType
        The eventstream for which the transition matrix is computed.

    See Also
    --------
    .Eventstream.transition_matrix : This method can be called on an Eventstream to obtain a TransitionMatrix.
    .TransitionGraph : An interactive tool for representing transitions as a graph.

    Notes
    -----
    For more detailed information, refer to the :doc:`Transition matrix user guide</user_guides/transition_matrix>`.
    """

    __eventstream: EventstreamType
    groups: UserGroupsType | None = None
    group_names: UserGroupsNamesType | None = None

    __nodelist: Nodelist
    __edgelist: Edgelist
    __values: pd.DataFrame
    __weight_col: str | None
    __norm_type: NormType
    __fill_value: Any
    __title: str

    def __init__(self, eventstream: EventstreamType) -> None:
        self._eventstream = eventstream
        self._nodelist = Nodelist(
            weight_cols=[eventstream.schema.event_id, *eventstream.schema.custom_cols],
            time_col=eventstream.schema.event_timestamp,
            event_col=eventstream.schema.event_name,
        )
        self._nodelist.calculate_nodelist(self._eventstream.to_dataframe())
        self._edgelist = Edgelist(eventstream=eventstream)
        self.__fill_value = 0
        self.__title = ""
        self.__norm_type = None
        self.__weight_col = None

    def fit(
        self,
        weight_col: Optional[str] = None,
        norm_type: NormType = None,
        groups: SplitExpr | None = None,
    ) -> None:
        """
        Calculates transition weights as a matrix for each unique pair of events.
        The calculation logic is the same that is used for edge weights calculation of transition graph.
        Applying ``fit`` method is necessary for the following usage
        of any visualization or descriptive ``TransitionMatrix`` methods.

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

        groups : tuple[list, list], tuple[str, str, str], str, optional
            Specify two groups of paths to plot differential transition matrix.
            Two transition matrices M1 and M2 will be calculated for these groups.
            Resulting matrix is M = M1 - M2.

            - If ``tuple[list, list]``, each sub-list should contain valid path ids.
            - If ``tuple[str, str, str]``, the first str should refer to a segment name,
              the others should refer to the corresponding segment values.
            - If ``str``, it should refer to a binary (i.e. containing two segment values only) segment name.

        """
        self.__weight_col = weight_col or self._eventstream.schema.user_id
        self.__norm_type = norm_type

        if groups:
            groups_, group_names = _split_segment(self._eventstream, groups)
            self.groups = groups_
            self.group_names = group_names

        if self.groups is None:
            self.__values = self._values(self.__weight_col, self.__norm_type)
            self.__title = "Transition matrix"
        else:
            event_list = list(self._nodelist.nodelist_df[self._nodelist.event_col])
            with pd.option_context("future.no_silent_downcasting", True):
                frame = pd.DataFrame(columns=event_list, index=event_list).fillna(self.__fill_value).infer_objects()
            positive_matrix = self._filter_group(self.groups[0])
            negative_matrix = self._filter_group(self.groups[1])
            self.__values = (
                frame.add(positive_matrix.__values, fill_value=self.__fill_value)
                .sub(negative_matrix.__values, fill_value=self.__fill_value)
                .fillna(self.__fill_value)
            )
            if self.groups:
                if self.group_names:
                    groups_subtitle = f", {self.group_names[0]} vs. {self.group_names[1]}"
                else:
                    groups_subtitle = ", group 1 vs. group 2"
            else:
                groups_subtitle = ""
            self.__title = (
                f"Differential transition matrix{groups_subtitle}\n"
                f"(group sizes: {positive_matrix._n_users}, {negative_matrix._n_users})"
            )

    def _filter_group(self, group: list) -> TransitionMatrix:
        if not isinstance(self._eventstream, FilterEventsHelperMixin):
            raise TypeError("filter_events is not implemented for the eventstream")
        substream = self._eventstream.filter_events(
            lambda df, schema: df[schema.user_id].isin(group)  # pyright: ignore [reportOptionalMemberAccess]
        )
        matrix = TransitionMatrix(substream)
        matrix.fit(weight_col=self.__weight_col, norm_type=self.__norm_type)
        return matrix

    def _values(self, weight_col: str, norm_type: NormType) -> pd.DataFrame:
        self._edgelist.calculate_edgelist(norm_type=norm_type, weight_cols=[weight_col])
        edgelist: pd.DataFrame = self._edgelist.edgelist_df.copy()
        edgelist = edgelist.drop(columns=["rete_is_out_of_threshold"])
        graph = nx.DiGraph()
        graph.add_weighted_edges_from(edgelist.values)

        return nx.to_pandas_adjacency(G=graph)

    @property
    def values(self) -> pd.DataFrame:
        """
        Returns the calculated transition matrix as a pandas.DataFrame.
        Should be used after :py:func:`fit`.
        """
        return self.__values.copy()

    @property
    def _n_users(self) -> int:
        return self._eventstream.to_dataframe()[self._eventstream.schema.user_id].nunique()

    def plot(
        self,
        heatmap_axis: Union[Literal["rows", "columns", "both"], int] = "both",
        precision: Union[int, Literal["auto"]] = "auto",
        figsize: Optional[Tuple[Union[float, int], Union[float, int]]] = None,
        show_large_matrix: Optional[bool] = None,
        show_values: Optional[bool] = None,
    ) -> Optional[matplotlib.axes.Axes]:
        """
        Create a heatmap plot based on the calculated transition matrix values.
        This method should be used after calling :py:func:`fit`.

        Parameters
        ----------
        heatmap_axis : {0 or 'rows', 1 or 'columns', 'both'}, default 'both'
            The axis for which the heatmap is to be generated.
            If specified, the heatmap will be created separately for the selected axis.
            If ``heatmap_axis='both'``, the heatmap will be applied to the entire matrix.

        figsize : tuple[float, float], default None
            The size of the visualization. The default size is calculated automatically depending
            on the matrix dimension and `precision` and `show_values` options.

        precision : int or str, default 'auto'
            The number of decimal digits to display after zero as fractions in the heatmap.
            If precision is ``auto``, the value will depend on the ``norm_type``:
            0 for ``norm_type=None``, and 2 otherwise.

        show_large_matrix : bool, optional
            If ``None`` the matrix is displayed only in case the matrix dimension <= 60.
            If ``True``, the matrix is plotted explicitly.

        show_values : bool, optional
            If ``None`` the matrix values are not displayed only in case the matrix dimension lies between 30 and 60.
            If ``True``, the matrix values are shown explicitly.
            If ``False``, the values are hidden, ``precision`` parameter is ignored in this case.

        Returns
        -------
        matplotlib.axes.Axes
            The Axes object containing the heatmap plot.

        """
        dim = self.__values.shape[0]

        if dim > MAX_DIM and not show_large_matrix:
            output = (
                f"The transition matrix has more than {MAX_DIM} events. We don't recommend to plot such large matrices,"
                + f" show_large_matrix=True or use the Sequences tool instead:\n"
                + f"{SEQUENCES_URL}\n"
                + "You can still get the matrix values as a pandas.DataFrame by retrieving the .values property:\n"
                + f"{TRANSITION_MATRIX_VALUES_URL}\n"
            )
            print(output)
            return None
        elif dim > SHOW_VALUES_DIM or show_values is False:
            dim_mode = "medium"
            annot = False
            linewidths = 0.01
            cbar = True
        else:
            dim_mode = "small"
            annot = self.__values
            linewidths = False
            cbar = False

        if precision == "auto":
            fmt = ".0f" if self.__norm_type is None else ".2f"
        # else precision is int
        else:
            fmt = f".{precision}f"

        if not figsize:
            # dim_mode == "medium"
            cell_size = 0.25
            if dim_mode == "small":
                cell_size = len(f"{self.__values.max().max():{fmt}}") * 0.05 + 0.4

            figsize = (round(self.values.shape[0] * cell_size), round(self.values.shape[0] * cell_size))

        grid_specs = {"wspace": 0.08, "hspace": 0.08}

        matrix = self._normalize(heatmap_axis)

        figure, axs = sns.mpl.pyplot.subplots(
            figsize=figsize,
            gridspec_kw=grid_specs,
        )

        heatmap = sns.heatmap(
            matrix, annot=annot, fmt=fmt, cmap="RdGy", center=0, cbar=cbar, linewidths=linewidths, linecolor="gray"
        )
        heatmap.set_title(self.__title, fontsize=16)

        sns.mpl.pyplot.sca(axs)
        sns.mpl.pyplot.yticks(rotation=0)
        return axs

    def _normalize(self, axis: Union[int, str]) -> pd.DataFrame:
        matrix = self.__values.copy()
        if axis == "both":
            return matrix
        if axis == 0 or axis == "rows":
            return matrix.div(matrix.abs().max(axis=1), axis=0).fillna(self.__fill_value)
        elif axis == 1 or axis == "columns":
            return matrix.div(matrix.abs().max(axis=0), axis=1).fillna(self.__fill_value)
        else:
            raise ValueError(f"no axis named {axis} for the transition matrix")


__all__ = ("TransitionMatrix",)
