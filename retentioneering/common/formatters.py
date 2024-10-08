from typing import Union

import pandas as pd


def fancy_round(df: pd.DataFrame) -> pd.DataFrame:
    """
    Format DataFrame numbers to a fancy str format.
    """

    def round_number(num: Union[float, int]) -> str:
        if isinstance(num, float):
            if abs(num) < 10:
                num = round(num, 3)
            elif abs(num) < 100:
                num = round(num, 2)
            else:
                num = round(num, 1)

        if isinstance(num, pd.Timedelta):
            seconds = num.total_seconds()  # type: ignore
            if abs(seconds) > 86400:
                return f"{round((seconds / 86400), 1)}d"
            elif abs(seconds) > 3600:
                return f"{round((seconds / 3600), 1)}h"
            elif abs(seconds) > 60:
                return f"{round((seconds / 60), 1)}m"
            elif abs(seconds) > 1:
                return f"{round(seconds, 1)}s"
            elif seconds == 0:
                return "0"
            else:
                return f"{round(seconds / 0.001, 1)}ms"

        if abs(num) >= 1000000000:
            return f"{round(num / 1000000000)}B"
        elif abs(num) >= 1000000:
            return f"{round(num / 1000000)}M"
        elif abs(num) >= 1000:
            return f"{round(num / 1000)}K"
        else:
            return str(num)

    rounded_df = df.map(round_number)  # type:ignore
    return rounded_df


def truncate_str(line: str, max_len: int) -> str:
    # a stupid workaround to include segment names that start with _ to the matplotlib legend
    # https://github.com/matplotlib/matplotlib/issues/21295
    line = " " + line

    if len(line) > max_len:
        return line[: max_len - 3] + "..."
    else:
        return line
