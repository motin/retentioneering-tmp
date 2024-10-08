from __future__ import annotations

import itertools
from typing import Callable, Tuple

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import colormaps, colors
from scipy.sparse import csc_matrix

from retentioneering.backend.tracker import collect_data_performance, time_performance
from retentioneering.common.constants import NGRAM_SEP, SEQUENCES_METRIC_TYPES
from retentioneering.eventstream.segments import _split_segment
from retentioneering.eventstream.types import (
    EventstreamType,
    SplitExpr,
    UserGroupsNamesType,
    UserGroupsType,
)


class Sequences:
    """
    A class that provides methods for patterns exploration.

    Parameters
    ----------
    eventstream : EventstreamType


    See Also
    --------
    .Eventstream.sequences : Call Sequences tool as an eventstream method.


    Notes
    -----
    See :doc:`Sequences user guide</user_guides/sequences>` for the details.
    """

    __eventstream: EventstreamType
    ngram_range: Tuple[int, int]
    groups: UserGroupsType | None
    path_id_col: str
    metrics: SEQUENCES_METRIC_TYPES | None
    final_columns_level_0: list[str]
    group_names: UserGroupsNamesType | None
    DEFAULT_GROUP_NAMES = ("group_1", "group_2")
    SEQUENCE_TYPE_GROUPS_COLUMN_NAME = ("sequence_type", "")
    SEQUENCE_TYPE_COLUMN_NAME = "sequence_type"
    RELATIVE_DELTA_COLUMN_NAME = "delta"
    ABS_DELTA_COLUMN_NAME = "delta_abs"
    ORIGINAL_METRICS_LIST = ["paths", "paths_share", "count", "count_share", "avg_count"]

    _vec_data: pd.DataFrame

    @time_performance(
        scope="sequences",
        event_name="init",
    )
    def __init__(self, eventstream: EventstreamType):
        self._default_columns_level_0: list = []
        self._sample_column_name: str
        self._full_metrics_list: list = []
        self._samples_full = None
        self.__eventstream = eventstream
        self.user_col = self.__eventstream.schema.user_id
        self.event_col = self.__eventstream.schema.event_name
        self.time_col = self.__eventstream.schema.event_timestamp
        self._sequence_df = pd.DataFrame()
        self._show_values_df = pd.DataFrame()
        self.groups = None

    def _calculate_metrics_df(self, vec_data: pd.DataFrame) -> pd.DataFrame:
        sequence_df = pd.concat(
            [vec_data.sum(), vec_data.astype(bool).sum(axis=0)], axis=1, keys=["count", self.path_id_col]
        ).astype(int)

        sequence_df["count_share"] = sequence_df["count"] / sequence_df["count"].sum()
        sequence_df[f"{self.path_id_col}_share"] = sequence_df[self.path_id_col] / len(vec_data)
        sequence_df["avg_count"] = sequence_df["count"] / sequence_df[self.path_id_col]
        sequence_df = sequence_df[self._full_metrics_list].fillna(0)

        return sequence_df

    def _add_support_set(self, vec_data: pd.DataFrame) -> pd.DataFrame:
        sparse_matrix = csc_matrix(vec_data.values)
        idx = vec_data.index
        samples_full = []
        for j, col in enumerate(vec_data.columns):
            non_zero_indices = sparse_matrix.indices[sparse_matrix.indptr[j] : sparse_matrix.indptr[j + 1]]
            samples_full.append([idx[non_zero_indices].tolist()])
        samples_full = pd.DataFrame(samples_full, index=vec_data.columns, columns=[self._sample_column_name])

        return samples_full

    @staticmethod
    def _sequence_type(x: str) -> str:
        temp = x.split(NGRAM_SEP)
        n = len(temp)
        n_unique = len(set(temp))
        if (n_unique > 1) and (temp[0] == temp[-1]):
            return "cycle"
        if (n_unique == 1) and (n > 1):
            return "loop"
        return "other"

    def _build_groups_df(
        self, vec_data: pd.DataFrame, groups: UserGroupsType, group_names: UserGroupsNamesType
    ) -> pd.DataFrame:
        vec_data["group"] = None
        vec_data.loc[groups[0], "group"] = group_names[0]
        vec_data.loc[groups[1], "group"] = group_names[1]
        group_mask = vec_data["group"].copy()
        vec_data.drop(columns="group", inplace=True)
        group_dfs = []
        samples_dfs = []
        for group in group_names:
            vec_data_group = vec_data[group_mask == group]
            sequence_df_group = self._calculate_metrics_df(vec_data_group)
            sequence_df_group.columns = pd.MultiIndex.from_product([sequence_df_group.columns, [group]])
            group_dfs.append(sequence_df_group)
            samples_full_group = self._add_support_set(vec_data=vec_data_group)
            samples_dfs.append(samples_full_group)

        sequence_df = pd.concat(group_dfs, axis=1)
        sequence_type = pd.Series(sequence_df.index).apply(lambda x: self._sequence_type(x))
        sequence_df[self.SEQUENCE_TYPE_GROUPS_COLUMN_NAME] = sequence_type.values

        samples_full = pd.concat(samples_dfs, axis=1)
        samples_columns = [(self._sample_column_name, group_names[0]), (self._sample_column_name, group_names[1])]
        samples_full.columns = pd.MultiIndex.from_tuples(samples_columns)

        for metric in self._full_metrics_list:
            sequence_df[(metric, self.ABS_DELTA_COLUMN_NAME)] = (
                sequence_df[(metric, group_names[1])] - sequence_df[(metric, group_names[0])]
            )
            sequence_df[(metric, self.RELATIVE_DELTA_COLUMN_NAME)] = (
                sequence_df[(metric, self.ABS_DELTA_COLUMN_NAME)] / sequence_df[(metric, group_names[0])]
            )

        final_columns = (
            list(itertools.product(self._full_metrics_list, self._default_columns_level_0))
            + [self.SEQUENCE_TYPE_GROUPS_COLUMN_NAME]
            + samples_columns
        )
        index_cols = pd.MultiIndex.from_tuples(final_columns)

        sequence_df = sequence_df.merge(samples_full, left_index=True, right_index=True)

        return sequence_df[index_cols]

    @time_performance(
        scope="sequences",
        event_name="fit",
    )
    def fit(
        self,
        ngram_range: Tuple[int, int] = (1, 1),
        groups: SplitExpr | None = None,
        group_names: UserGroupsNamesType | None = None,
        path_id_col: str | None = None,
    ) -> None:
        """
        Calculate statistics on n-grams found in eventstream.
        Calculated path_metrics:

        - ``paths``: The number of unique paths that contain each particular event sequence
          (calculated within the specified ``path_id_col``, or ``user_id`` by default).
        - ``paths_share``: The ratio of paths to the sum of paths.
        - ``count``: The number of occurrences of a particular sequence.
        - ``count_share``: The ratio of count to the sum of counts.
        - ``avg_count``: The average number of occurrences per path.

        Defined sequences types:

        - ``loop`` - if sequence length >= 2 and all the events are equal.
        - ``cycle`` - if sequence length >= 3 and start and end events are equal.
        - ``other`` - all other sequences.

        Parameters
        ----------
        ngram_range : Tuple(int, int)
            The lower and upper boundary of the range of n-values for different word n-grams to be
            extracted. For example, ngram_range=(1, 1) means only single events, (1, 2) means single events
            and bigrams.
        groups : SplitExpr, optional
            Can be specified to calculate statistics for n-grams group-wise and provide delta values.
            Must contain a tuple of two elements (g_1, g_2), where g_1 and g_2 are collections
            of path IDs (for the column specified in the ``path_id_col`` parameter).
        group_names : UserGroupsNamesType, optional
            Names for the selected groups g_1 and g_2, which will be shown in the final plot header.
        path_id_col : str, optional
            The column used for calculating 'paths' and 'paths_share' path_metrics.
            If not specified, the ``user_id`` from ``eventstream.schema``
            will be used. For example, it can be specified as ``session_id`` if ``eventstream``
            has such a ``custom_col``.


        Returns
        -------
        None

        Notes
        -----
        See the results of calculation using :py:func:`plot` method and the :py:func:`values` attribute.

        """
        called_params = {
            "ngram_range": ngram_range,
            "groups": groups,
            "group_names": group_names,
            "path_id_col": path_id_col,
        }
        not_hash_values = ["ngram_range"]

        self.ngram_range = ngram_range
        self.path_id_col = path_id_col or self.__eventstream.schema.user_id
        self._sample_column_name = f"{self.path_id_col}_sample"
        self._full_metrics_list = [self.path_id_col, f"{self.path_id_col}_share", "count", "count_share", "avg_count"]

        if groups:
            self.groups, _ = _split_segment(self.__eventstream, groups)

        if group_names:
            self.group_names = group_names

        else:
            self.group_names = self.DEFAULT_GROUP_NAMES

        vec_data = self.__eventstream.extract_features(
            feature_type="count", ngram_range=self.ngram_range, path_id_col=self.path_id_col
        )

        if self.groups:
            self._default_columns_level_0 = [
                self.group_names[0],
                self.group_names[1],
                self.ABS_DELTA_COLUMN_NAME,
                self.RELATIVE_DELTA_COLUMN_NAME,
            ]
            vec_data = vec_data[vec_data.index.isin(list(self.groups[0]) + list(self.groups[1]))]  # type: ignore

        if self.groups:
            self._sequence_df = self._build_groups_df(
                vec_data=vec_data, groups=self.groups, group_names=self.group_names
            )

        else:
            sequence_df = self._calculate_metrics_df(vec_data)

            sequence_type = pd.Series(sequence_df.index).apply(lambda x: self._sequence_type(x))
            sequence_df[self.SEQUENCE_TYPE_COLUMN_NAME] = sequence_type.values
            samples_full = self._add_support_set(vec_data=vec_data)
            self._sequence_df = sequence_df.join(samples_full[self._sample_column_name])

        self._sequence_df.index.name = "Sequence"
        self._show_values_df = self._sequence_df.copy()

        collect_data_performance(
            scope="sequences",
            event_name="metadata",
            called_params=called_params,
            not_hash_values=not_hash_values,
            performance_data={"shape": self._show_values_df.shape},
            eventstream_index=self.__eventstream._eventstream_index,
        )

    def _size_sample_ids(
        self, data: pd.DataFrame, sample_size: int | None, final_columns_level_0: list
    ) -> pd.DataFrame:
        if sample_size:
            displayed_columns = final_columns_level_0 + [self.SEQUENCE_TYPE_COLUMN_NAME] + [self._sample_column_name]
            data = data[displayed_columns].copy()

            def shuffle_and_slice(x: pd.Series, size: int = sample_size) -> pd.Series:  # type: ignore
                np.random.seed(42)
                np.random.shuffle(x)
                return x[:size]

            if self.groups:
                for group in self.group_names:  # type: ignore
                    data[(self._sample_column_name, group)] = data[(self._sample_column_name, group)].apply(
                        lambda x: shuffle_and_slice(x)
                    )
            else:
                data[self._sample_column_name] = data[self._sample_column_name].apply(lambda x: shuffle_and_slice(x))  # type: ignore

        else:
            displayed_columns = final_columns_level_0 + [self.SEQUENCE_TYPE_COLUMN_NAME]
            data = self._sequence_df[displayed_columns]

        return data

    @staticmethod
    def _create_plot(
        data: pd.DataFrame, heatmap_columns: list, precision: int | None
    ) -> pd.io.formats.style.Styler:  # type: ignore
        columns_intersection = set(data.columns) & set(heatmap_columns)
        if not columns_intersection:
            return

        heatmap_values = data.copy(deep=True)
        numeric_cols = data.select_dtypes(include="number").columns
        max_values = data[numeric_cols].replace(+np.inf, -np.inf).max()
        min_values = data[numeric_cols].replace(-np.inf, +np.inf).min()
        heatmap_values[numeric_cols] = heatmap_values[numeric_cols].replace(+np.inf, max_values)  # type: ignore
        heatmap_values[numeric_cols] = heatmap_values[numeric_cols].replace(-np.inf, min_values)  # type: ignore

        cm = sns.diverging_palette(h_neg=246, h_pos=27, s=99, l=68, as_cmap=True)

        def add_background(s: pd.Series, cmap: str = "PuBu") -> list[str]:  # type: ignore
            a = heatmap_values.loc[:, s.name].copy()  # type: ignore

            if a.min() >= 0:
                norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=a.max())

            elif a.max() <= 0:
                norm = mcolors.TwoSlopeNorm(vmin=a.min(), vcenter=0, vmax=1)

            else:
                norm = mcolors.TwoSlopeNorm(vmin=a.min(), vcenter=0.0, vmax=a.max())

            normed = norm(a.values)
            c = [colors.rgb2hex(x) for x in colormaps.get_cmap(cmap)(normed)]
            return ["background-color: %s" % color for color in c]

        plot_sequence_df_styler = data.style.apply(add_background, subset=heatmap_columns, cmap=cm).format(
            precision=precision, thousands=" "
        )
        return plot_sequence_df_styler

    def _check_params(
        self,
        plot_sequence_df: pd.DataFrame,
        metrics: SEQUENCES_METRIC_TYPES | None = None,
        threshold: tuple[str, float | int] | None = None,
        sorting: tuple[str | tuple, bool] | tuple[list[str | tuple], list[bool]] | None = None,
        heatmap_columns: str | list[str | tuple] | None = None,
    ) -> Tuple[list, tuple, tuple, list]:
        replacements = {"paths": self.path_id_col, "paths_share": f"{self.path_id_col}_share"}
        replacer = replacements.get

        final_columns_full, final_columns_level_0 = self._check_metrics(
            plot_sequence_df=plot_sequence_df, metrics=metrics, replacer=replacer
        )

        heatmap_columns_list: list
        heatmap_columns_list = self._define_heatmap_columns(
            heatmap_columns=heatmap_columns,
            replacer=replacer,
            final_columns_level_0=final_columns_level_0,
            final_columns_full=final_columns_full,
        )

        sorting_checked = self._define_sorting_columns(
            sorting=sorting, replacer=replacer, final_columns_full=final_columns_full
        )

        threshold_checked: Tuple[list | str | None, float | int | None]
        if threshold:
            threshold_checked = self._define_threshold_columns(
                threshold=threshold, replacer=replacer, final_columns_full=final_columns_full
            )
        else:
            threshold_checked = None, None

        return heatmap_columns_list, sorting_checked, threshold_checked, final_columns_level_0

    def _check_metrics(
        self, plot_sequence_df: pd.DataFrame, replacer: Callable, metrics: SEQUENCES_METRIC_TYPES | None = None
    ) -> Tuple[list, list]:
        if self.groups:
            final_columns_level_0 = [self.path_id_col]
        else:
            final_columns_level_0 = self._full_metrics_list

        if metrics:
            if isinstance(metrics, str):  # type: ignore
                final_metrics = [metrics]
            else:
                final_metrics = metrics
            if not set(final_metrics).issubset(set(self.ORIGINAL_METRICS_LIST)):
                unexpected_metrics = set(final_metrics).difference(set(self.ORIGINAL_METRICS_LIST))
                raise ValueError(
                    f"""Unexpected path_metrics value: {unexpected_metrics}. Should be a metric name or
                                 a list of metric names from {self.ORIGINAL_METRICS_LIST}."""
                )

            final_columns_level_0 = [replacer(n, n) for n in final_metrics]
        final_columns_full = list(plot_sequence_df[final_columns_level_0].columns)
        return final_columns_full, final_columns_level_0

    def _define_sorting_columns(
        self,
        sorting: tuple[str | tuple, bool] | tuple[list[str | tuple], list[bool]] | None,
        replacer: Callable,
        final_columns_full: list,
    ) -> Tuple[str | list | tuple, bool | list | tuple]:
        # sorting
        if sorting:
            error_message_text_sorting = f"""Should be tuple or list with two elements:
            (column_name, bool) or ([column_name], [list of bool]). Note that those two elements should be the
            same length. For example: ('count_share', False), ([('count_share', 'delta'),('paths_share', 'delta')],
            [False, True])."""
            sorting_columns: str | list | tuple
            sorting_direction: bool | list | tuple
            if not self.groups:
                if isinstance(sorting, (list, tuple)):  # type: ignore
                    if isinstance(sorting[0], str) and isinstance(sorting[1], bool):
                        sorting_columns = [sorting[0]]
                        sorting_direction = [sorting[1]]

                    elif len(sorting) == 2 and all(isinstance(elem, (list, tuple)) for elem in sorting):
                        sorting_columns = sorting[0]
                        sorting_direction = sorting[1]
                        if (
                            not all(isinstance(elem, str) for elem in sorting_columns)
                            or not all(isinstance(elem, bool) for elem in sorting_direction)  # type: ignore
                            or not len(sorting_columns) == len(sorting_direction)  # type: ignore
                        ):
                            raise TypeError(f"Unexpected sorting parameter type. {error_message_text_sorting}")
                    else:
                        raise TypeError(f"Unexpected sorting parameter type. {error_message_text_sorting}")
                else:
                    raise TypeError(f"Unexpected sorting parameter type. {error_message_text_sorting}")

                sorting_columns = [(replacer(col, col)) for col in sorting_columns]

            else:
                if isinstance(sorting, (list, tuple)):  # type: ignore
                    if isinstance(sorting[0], (list, tuple)) and isinstance(sorting[1], bool):
                        sorting_columns = [sorting[0]]
                        sorting_direction = [sorting[1]]

                    elif len(sorting) == 2 and all(isinstance(elem, (list, tuple)) for elem in sorting):
                        sorting_columns = sorting[0]
                        sorting_direction = sorting[1]
                        if len(sorting_columns) != len(sorting_direction):  # type: ignore
                            raise TypeError(f"Unexpected sorting parameter type. {error_message_text_sorting}")
                    else:
                        raise TypeError(f"Unexpected sorting parameter type. {error_message_text_sorting}")

                sorting_columns = [(replacer(col[0], col[0]), col[1]) for col in sorting_columns]

            if not set(sorting_columns).issubset(set(final_columns_full)):
                unexpected_sorting_columns = set(sorting_columns).difference(set(final_columns_full))
                raise ValueError(
                    f"""Unexpected sorting_columns name: {unexpected_sorting_columns}.
                                     {error_message_text_sorting}"""
                )
        else:
            if self.groups:
                sorting_columns = final_columns_full[3]
                sorting_direction = False
            else:
                sorting_columns = final_columns_full[0]
                sorting_direction = False

        return sorting_columns, sorting_direction

    def _define_threshold_columns(
        self, threshold: tuple[str | list, float | int], replacer: Callable, final_columns_full: list
    ) -> Tuple[list | str | None, float | int | None]:
        threshold_column, threshold_value = threshold
        error_message_text_thresh = f"""Should be a tuple or list with column_name as the first element and int or
                                        float values as the second.
                                        For example: ('count_share', 0.5), (('count_share', 'delta'), 0.5)."""
        if not isinstance(threshold_value, (int, float)):  # type: ignore
            raise TypeError(f"Unexpected threshold parameter type. {error_message_text_thresh}")

        if not self.groups:
            if not isinstance(threshold_column, str):  # type: ignore
                raise TypeError(f"Unexpected threshold parameter type. {error_message_text_thresh}")
            else:
                threshold_column = replacer(threshold_column, threshold_column)

        else:
            if not isinstance(threshold_column, (list, tuple)):
                raise TypeError(f"Unexpected threshold parameter type. {error_message_text_thresh}")
            elif not len(threshold_column) == 2 and all(isinstance(elem, str) for elem in threshold_column):  # type: ignore
                raise TypeError(f"Unexpected threshold parameter type. {error_message_text_thresh}")
            else:
                threshold_column = (replacer(threshold_column[0], threshold_column[0]), threshold_column[1])  # type: ignore

        if threshold_column not in final_columns_full:
            raise ValueError(
                f"""Unexpected sorting_columns name: {threshold_column}.
                                                 {error_message_text_thresh}"""
            )

        return threshold_column, threshold_value  # type: ignore

    def _define_heatmap_columns(
        self,
        heatmap_columns: str | list[str | tuple] | None,
        replacer: Callable,
        final_columns_level_0: list,
        final_columns_full: list,
    ) -> list:
        error_message_text_heatmap = f"""Should be a column name or a list
                                        of column names. For example: ['count_share', 'delta'],
                                        [('count_share', 'delta'),('paths_share', 'delta')]."""
        if heatmap_columns:
            if not self.groups:
                if isinstance(heatmap_columns, str):
                    heatmap_columns = [heatmap_columns]
                elif isinstance(heatmap_columns, (list, tuple)):  # type: ignore
                    if not all(isinstance(col, str) for col in heatmap_columns):
                        raise TypeError(f"Unexpected heatmap_columns type. {error_message_text_heatmap}")
                else:
                    raise TypeError(f"Unexpected heatmap_columns type. {error_message_text_heatmap}")

                heatmap_columns = [replacer(col, col) for col in heatmap_columns]

            else:
                if isinstance(heatmap_columns, (list, tuple)):
                    if len(heatmap_columns) == 2 and all(isinstance(col, str) for col in heatmap_columns):
                        heatmap_columns = [heatmap_columns]  # type: ignore
                    elif not all(isinstance(col, (list, tuple)) for col in heatmap_columns):
                        raise TypeError(f"Unexpected heatmap_columns type. {error_message_text_heatmap}")

                    heatmap_columns = [(replacer(col[0], col[0]), col[1]) for col in heatmap_columns]  # type: ignore

            if not set(heatmap_columns).issubset(set(final_columns_full)):
                unexpected_heatmap_columns = set(heatmap_columns).difference(set(final_columns_full))
                raise ValueError(
                    f"""Unexpected heatmap_columns name: {unexpected_heatmap_columns}.
                                            {error_message_text_heatmap}"""
                )

        else:
            if self.groups:
                heatmap_columns = list(itertools.product(final_columns_level_0, [self.RELATIVE_DELTA_COLUMN_NAME]))
            else:
                heatmap_columns = [f"{self.path_id_col}_share"]
        return heatmap_columns  # type: ignore

    @time_performance(
        scope="sequences",
        event_name="plot",
    )
    def plot(
        self,
        metrics: SEQUENCES_METRIC_TYPES | None = None,
        threshold: tuple[str, float | int] | None = None,
        sorting: tuple[str | tuple, bool] | tuple[list[str | tuple], list[bool]] | None = None,
        heatmap_cols: str | list[str | tuple] | None = None,
        sample_size: int | None = 1,
        precision: int = 2,
    ) -> pd.io.formats.style.Styler:  # type: ignore
        """
        Parameters
        ----------
        metrics : {'paths', 'paths_share', 'count', 'count_share'}, optional
            Specify the path_metrics to be displayed in the plot.

            - If groups are specified, by default, only the 'paths' metric will be plotted within each group,
              along with both deltas (relative and absolute).
            - If ``groups=None``, all four path_metrics will be shown by default.

        threshold : tuple[str, float | int], optional
            Used to filter out infrequent sequences based on the specified metric.

            - Example without groups: ('paths', 1200)
            - Example with groups: (('user_id', 'group_1'), 1200)

            Only rows with values greater than or equal to 1200 in the specified column will be displayed.

        sorting : Tuple(str or list of str, bool or list of bool) or None, default None
            - The first element in the tuple: Column name or list of names for sorting.
            - The second element: The sorting order (ascending vs. descending). Specify a list for multiple
              sorting orders.
              If a list of bools is provided, it must match the length of the sorting columns.

        heatmap_cols : str or list of str or list of tuples or None
            Specifies columns to be represented in the heatmap as follows:

            - The heatmap range is calculated column-wise.
            - For columns containing negative values, the palette will be divergent (blue - orange with white as zero).
            - For columns with only positive values, the palette will be orange.
            - For columns with only negative values, the palette will be blue.

            Default values

        sample_size : int or None, default 1
            Number of ID samples to display.
        precision : int, default 2
            Number of decimal digits to show as fractions in the heatmap.

        Returns
        -------
        pd.io.formats.style.Styler
            Styled pd.Dataframe object.

        """
        called_params = {
            "path_metrics": metrics,
            "threshold": threshold,
            "sorting": sorting,
            "heatmap_cols": heatmap_cols,
            "sample_size": sample_size,
            "precision": precision,
        }

        not_hash_values = ["path_metrics"]

        plot_sequence_df = self._sequence_df.copy()
        heatmap_columns_list, sorting_checked, threshold_checked, final_columns_level_0 = self._check_params(
            plot_sequence_df, metrics, threshold, sorting, heatmap_cols
        )

        sorting_columns, sorting_direction = sorting_checked
        threshold_column, threshold_value = threshold_checked
        plot_sequence_df = self._size_sample_ids(
            data=plot_sequence_df, sample_size=sample_size, final_columns_level_0=final_columns_level_0
        )

        plot_sequence_df = plot_sequence_df.sort_values(by=sorting_columns, ascending=sorting_direction)
        if threshold:
            plot_sequence_df = plot_sequence_df[plot_sequence_df[threshold_column] >= abs(threshold_value)]

        if precision:
            plot_sequence_df = plot_sequence_df.round(precision)

        plot_sequence_df_styler = self._create_plot(
            data=plot_sequence_df, heatmap_columns=heatmap_columns_list, precision=precision
        )

        self._show_values_df = plot_sequence_df
        collect_data_performance(
            scope="sequences",
            event_name="metadata",
            called_params=called_params,
            not_hash_values=not_hash_values,
            performance_data={"shape": plot_sequence_df.shape},
            eventstream_index=self.__eventstream._eventstream_index,
        )

        return plot_sequence_df_styler

    @property
    @time_performance(
        scope="sequences",
        event_name="values",
    )
    def values(self) -> pd.DataFrame:
        """
        Returns a pd.DataFrame representing the fitted or plotted Sequences table.
        Should be used after :py:func:`fit` or :py:func:`plot`.


        Returns
        -------
        pd.DataFrame



        """
        return self._show_values_df

    @property
    @time_performance(
        scope="sequences",
        event_name="params",
    )
    def params(self) -> dict[str, tuple | None | str | UserGroupsType | UserGroupsNamesType]:
        """
        Returns the parameters used for the last fitting.
        Should be used after :py:func:`fit`.

        """
        return {
            "ngram_range": self.ngram_range,
            "groups": self.groups,
            "group_names": self.group_names,
            "path_id": self.path_id_col,
        }
