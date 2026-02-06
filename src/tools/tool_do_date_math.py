import datetime
import dateutil

from typing import Literal
from pydantic import BaseModel, Field

from langchain.tools import tool


def today_xml():
    today = datetime.date.today()
    return (
        f"Today is "
        f"<today_date>{today.isoformat()}"
        f"<day>{today.day}</day>"
        f"<month>{today.month}</month>"
        f"<year>{today.year}</year>"
        f"</today_date>"
    )


class DateMathInput(BaseModel):
    base_date: str = Field(description="Base date in the format YYYY-MM-DD")
    deltas: list[int] = Field(
        description="Intervals, as defined in delta_type, to add or subtract from the base date"
    )
    delta_type: Literal["day", "week", "month", "year"] = Field(
        description="Type of interval to sum or subtract from base_date. Possible values are: ['day', 'week', 'month', 'year']"
    )


@tool(args_schema=DateMathInput)
def do_date_math(base_date, deltas, delta_type) -> list[str]:
    """Adds or subtracts one or more time intervals from a given date in the format YYYY-MM-DD.
    The <deltas></deltas> to be added or subtracted should be separated by commas. Use negative values to subtract, as shown in the <example_deltas></example_deltas>:

    <example_deltas>
    <example_delta>5<example_delta>
    <example_delta>-7, -14, -21<example_delta>
    <example_delta>5, -6, -8<example_delta>
    </example_deltas>

        Raises ValueError: if one of the parameters is invalid."""

    allowed_delta_types = ["day", "week", "month", "year"]
    if delta_type not in allowed_delta_types:
        return f"Error: delta_type must be one of {allowed_delta_types}"

    try:
        date_object = datetime.datetime.strptime(base_date, "%Y-%m-%d").date()
    except Exception as e:
        return f"Error: please provide a base_date in the format YYYY-MM-DD. Error: {e}"

    delta_periods = deltas

    if delta_type == "day":
        final_deltas = [
            dateutil.relativedelta.relativedelta(days=x) for x in delta_periods
        ]
    elif delta_type == "week":
        final_deltas = [
            dateutil.relativedelta.relativedelta(weeks=x) for x in delta_periods
        ]
    elif delta_type == "month":
        final_deltas = [
            dateutil.relativedelta.relativedelta(months=x) for x in delta_periods
        ]
    elif delta_type == "year":
        final_deltas = [
            dateutil.relativedelta.relativedelta(years=x) for x in delta_periods
        ]

    ans = [(date_object + x).strftime("%Y-%m-%d %A") for x in final_deltas]
    return ans
