import base64
import io
import urllib
from typing import Callable, Optional

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import wasserstein_distance

from retentioneering.common.formatters import truncate_str
from retentioneering.eventstream.segments import _split_segment
from retentioneering.eventstream.types import EventstreamType, SplitExpr


class SegmentDiff:
    def __init__(self, eventstream: EventstreamType):
        self.__eventstream = eventstream

    def fit(
        self,
        features: pd.DataFrame,
        segment_items: SplitExpr,
        aggfunc: Callable = np.mean,
        threshold: float = 0.01,
        top_n: Optional[int] = None,
    ) -> pd.DataFrame:
        from retentioneering.eventstream.eventstream import Eventstream

        stream: Eventstream = self.__eventstream  # type: ignore
        segment_name, segment_value_1, segment_value_2 = segment_items
        stream.materialize_segment(segment_name)

        user_groups, user_group_names = _split_segment(stream, segment_items)

        segment_value_1_users, segment_value_2_users = user_groups
        features_1 = features.loc[segment_value_1_users].assign(segment_value=segment_value_1)
        features_2 = features.loc[segment_value_2_users].assign(segment_value=segment_value_2)

        if len(features_1) < 10:
            raise ValueError(
                f"Segment {segment_name}={segment_value_1} contains less than 10 paths."
                "The density diagrams will be meaningless."
            )

        if len(features_2) < 10:
            raise ValueError(
                f"Segment {segment_name}={segment_value_2} contains less than 10 paths."
                "The density diagrams will be meaningless"
            )

        segment_name_1 = truncate_str(segment_value_1, max_len=20)
        segment_name_2 = truncate_str(segment_value_2, max_len=20)

        figsize = (4, 0.7)
        right_adjustment = max(0.5, 0.7 - (max(len(segment_name_1), len(segment_name_2)) / 20) * 0.3)

        res_list = []
        for feature in features.columns:
            f1 = features_1[feature]
            f2 = features_2[feature]
            f1_agg = f1.pipe(aggfunc)
            f2_agg = f2.pipe(aggfunc)
            wd = wasserstein_distance(f1, f2)

            if wd > threshold:
                fig, ax = plt.subplots(figsize=figsize)

                if f1.nunique() > 1:
                    sns.kdeplot(x=f1, bw_method=0.3, ax=ax, label=segment_name_1, color="C0", warn_singular=False)
                    ax.axvline(x=f1_agg, linestyle="--", color="C0")
                else:
                    sns.barplot(x=f1, ax=ax)
                if f2.nunique() > 1:
                    sns.kdeplot(x=f2, bw_method=0.3, ax=ax, label=segment_name_2, color="C1", warn_singular=False)
                    ax.axvline(x=f2_agg, linestyle="--", color="C1")
                else:
                    sns.barplot(x=f2, ax=ax)

                ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))

                ax.yaxis.set_visible(False)
                ax.set_xlabel("")
                ax.margins(x=0, y=0)
                plt.subplots_adjust(left=0, right=right_adjustment, top=1, bottom=0.3)

                # convert plt output to base64 and insert it as <img> html tag.
                buf = io.BytesIO()
                plt.savefig(buf, format="png")
                plt.close()
                buf.seek(0)
                img_string = base64.b64encode(buf.read())
                img_uri = "data:image/png;base64," + urllib.parse.quote(img_string)  # type: ignore
                img_html = '<img src = "%s"/>' % img_uri
            else:
                img_html = ""

            res_list.append([feature, f1_agg, f2_agg, wd, img_html])

        mertic_1_col = f"{segment_name}={segment_name_1}"
        mertic_2_col = f"{segment_name}={segment_name_2}"

        res = (
            pd.DataFrame(res_list, columns=["feature", mertic_1_col, mertic_2_col, "distance", "feature distribution"])
            .sort_values("distance", ascending=False)
            .reset_index(drop=True)
        )

        if top_n is not None:
            res = res.head(top_n)
        else:
            res = res[res["distance"] > threshold]

        return res
