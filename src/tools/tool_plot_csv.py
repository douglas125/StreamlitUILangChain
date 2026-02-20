from __future__ import annotations

import datetime as dt
import decimal
import re
import uuid
from pathlib import Path
from typing import Any
from typing import Literal

from pydantic import BaseModel
from pydantic import Field

from langchain.tools import tool

CSV_TABLE_NAME = "csv_data"
DEFAULT_MAX_POINTS = 2000
MAX_ALLOWED_POINTS = 10000
MEDIA_DIR = Path(__file__).resolve().parents[2] / "media"
FORBIDDEN_SQL_PATTERN = re.compile(
    r"\b("
    r"insert|update|delete|create|alter|drop|truncate|copy|attach|detach|"
    r"call|pragma|vacuum|replace|merge|grant|revoke"
    r")\b",
    re.IGNORECASE,
)


class PlotCsvInput(BaseModel):
    file_path: str = Field(
        description="Local path to a CSV file that will be exposed as table 'csv_data'."
    )
    query: str = Field(
        description=(
            "SQL query to execute against table 'csv_data'. "
            "Use SELECT-style queries."
        )
    )
    chart_type: Literal["bar", "line", "scatter", "area"] = Field(
        description="Chart type to render as static image."
    )
    x: str = Field(description="Column name used for x axis.")
    y: str = Field(description="Column name used for y axis.")
    color: str | None = Field(
        default=None,
        description="Optional column name for color grouping.",
    )
    title: str | None = Field(default=None, description="Optional chart title.")
    max_points: int = Field(
        default=DEFAULT_MAX_POINTS,
        ge=10,
        le=MAX_ALLOWED_POINTS,
        description="Maximum number of rows included in the visualization.",
    )


class MediaContent(BaseModel):
    type: Literal["image"] = Field(description="Media type for Streamlit rendering.")
    url: str = Field(description="Local file path of generated plot image.")


class PlotCsvOutput(BaseModel):
    media_content: MediaContent = Field(description="Generated chart image payload.")
    image_file: str = Field(description="Generated image file name in media folder.")
    table_name: str = Field(description="Table name available in SQL.")
    columns: list[str] = Field(description="Column names returned by the query.")
    row_count: int = Field(description="Number of rows used in the chart.")
    applied_limit: int = Field(description="Row cap enforced by the tool.")


def _normalize_sql(query: str) -> str:
    statements = [statement.strip() for statement in query.split(";") if statement.strip()]
    if len(statements) == 0:
        raise ValueError("Query cannot be empty.")
    if len(statements) != 1:
        raise ValueError("Provide exactly one SQL statement per call.")
    statement = statements[0]
    lowered = statement.lower()
    if not (lowered.startswith("select") or lowered.startswith("with")):
        raise ValueError("Only SELECT-style queries are allowed.")
    if FORBIDDEN_SQL_PATTERN.search(statement):
        raise ValueError("Only read-only SQL is allowed for this tool.")
    return statement


def _to_json_safe(value: Any) -> Any:
    if isinstance(value, dt.datetime):
        return value.isoformat()
    if isinstance(value, dt.date):
        return value.isoformat()
    if isinstance(value, dt.time):
        return value.isoformat()
    if isinstance(value, dt.timedelta):
        return value.total_seconds()
    if isinstance(value, decimal.Decimal):
        return float(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, list):
        return [_to_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_to_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_json_safe(inner_value) for key, inner_value in value.items()}
    return value


def _sql_quote(value: str) -> str:
    return value.replace("'", "''")


def _coerce_numeric(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, decimal.Decimal):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip().replace("R$", "").replace("$", "").replace(" ", "")
        if not text:
            return None
        if "," in text and "." in text:
            if text.rfind(",") > text.rfind("."):
                text = text.replace(".", "").replace(",", ".")
            else:
                text = text.replace(",", "")
        elif "," in text:
            parts = text.split(",")
            if len(parts) == 2 and len(parts[1]) <= 2:
                text = text.replace(",", ".")
            else:
                text = text.replace(",", "")
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _extract_points(
    rows_raw: list[tuple[Any, ...]],
    columns: list[str],
    x: str,
    y: str,
    color: str | None,
) -> list[dict[str, Any]]:
    x_idx = columns.index(x)
    y_idx = columns.index(y)
    color_idx = None
    if color is None:
        color_idx = None
    else:
        color_idx = columns.index(color)

    points: list[dict[str, Any]] = []
    for row in rows_raw:
        x_value = row[x_idx]
        y_value = _coerce_numeric(row[y_idx])
        if x_value is None or y_value is None:
            pass
        else:
            color_value = None
            if color_idx is None:
                color_value = None
            else:
                color_value = row[color_idx]
            points.append(
                {
                    "x": _to_json_safe(x_value),
                    "y": y_value,
                    "color": _to_json_safe(color_value),
                }
            )
    return points


def _set_discrete_xticks(ax, labels: list[str]) -> None:
    if len(labels) == 0:
        return
    positions = list(range(len(labels)))
    if len(labels) <= 25:
        tick_positions = positions
    else:
        step = max(1, len(labels) // 12)
        tick_positions = list(range(0, len(labels), step))
        if tick_positions[-1] != len(labels) - 1:
            tick_positions.append(len(labels) - 1)
    tick_labels = [labels[pos] for pos in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")


def _render_static_plot_image(
    points: list[dict[str, Any]],
    chart_type: str,
    x_label: str,
    y_label: str,
    color_label: str | None,
    title: str | None,
) -> Path:
    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except ModuleNotFoundError as exc:
        raise ValueError(
            "matplotlib is required for tool_plot_csv static image generation."
        ) from exc

    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    filename = (
        f"plot_{dt.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_"
        f"{uuid.uuid4().hex[:8]}.png"
    )
    output_path = MEDIA_DIR / filename

    fig, ax = plt.subplots(figsize=(11, 5.5))
    positions = list(range(len(points)))
    labels = [str(point["x"]) for point in points]
    y_values = [float(point["y"]) for point in points]

    if color_label:
        unique_groups: list[str] = []
        for point in points:
            group = str(point["color"]) if point["color"] is not None else "Unknown"
            if group in unique_groups:
                pass
            else:
                unique_groups.append(group)
        cmap = plt.get_cmap("tab20")
        group_to_color: dict[str, Any] = {}
        for idx, group in enumerate(unique_groups):
            group_to_color[group] = cmap(idx % 20)
    else:
        unique_groups = []
        group_to_color = {}

    if chart_type == "bar":
        if color_label:
            for group in unique_groups:
                group_positions = []
                group_values = []
                for idx, point in enumerate(points):
                    current_group = (
                        str(point["color"]) if point["color"] is not None else "Unknown"
                    )
                    if current_group == group:
                        group_positions.append(idx)
                        group_values.append(float(point["y"]))
                ax.bar(
                    group_positions,
                    group_values,
                    color=group_to_color[group],
                    label=group,
                    alpha=0.9,
                )
            if len(unique_groups) > 1:
                ax.legend(title=color_label)
        else:
            ax.bar(positions, y_values, color="#4C78A8", alpha=0.9)
    elif chart_type == "line":
        if color_label:
            for group in unique_groups:
                group_positions = []
                group_values = []
                for idx, point in enumerate(points):
                    current_group = (
                        str(point["color"]) if point["color"] is not None else "Unknown"
                    )
                    if current_group == group:
                        group_positions.append(idx)
                        group_values.append(float(point["y"]))
                ax.plot(
                    group_positions,
                    group_values,
                    marker="o",
                    linewidth=2,
                    label=group,
                    color=group_to_color[group],
                )
            if len(unique_groups) > 1:
                ax.legend(title=color_label)
        else:
            ax.plot(positions, y_values, marker="o", linewidth=2, color="#4C78A8")
    elif chart_type == "scatter":
        if color_label:
            for group in unique_groups:
                group_positions = []
                group_values = []
                for idx, point in enumerate(points):
                    current_group = (
                        str(point["color"]) if point["color"] is not None else "Unknown"
                    )
                    if current_group == group:
                        group_positions.append(idx)
                        group_values.append(float(point["y"]))
                ax.scatter(
                    group_positions,
                    group_values,
                    label=group,
                    s=36,
                    color=group_to_color[group],
                    alpha=0.85,
                )
            if len(unique_groups) > 1:
                ax.legend(title=color_label)
        else:
            ax.scatter(positions, y_values, s=36, color="#4C78A8", alpha=0.85)
    else:
        if color_label:
            for group in unique_groups:
                group_positions = []
                group_values = []
                for idx, point in enumerate(points):
                    current_group = (
                        str(point["color"]) if point["color"] is not None else "Unknown"
                    )
                    if current_group == group:
                        group_positions.append(idx)
                        group_values.append(float(point["y"]))
                ax.plot(
                    group_positions,
                    group_values,
                    linewidth=2,
                    color=group_to_color[group],
                    label=group,
                )
                ax.fill_between(
                    group_positions,
                    group_values,
                    alpha=0.2,
                    color=group_to_color[group],
                )
            if len(unique_groups) > 1:
                ax.legend(title=color_label)
        else:
            ax.plot(positions, y_values, linewidth=2, color="#4C78A8")
            ax.fill_between(positions, y_values, alpha=0.25, color="#4C78A8")

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"{chart_type.capitalize()} chart")
    _set_discrete_xticks(ax, labels)
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


@tool(args_schema=PlotCsvInput)
def tool_plot_csv(
    file_path: str,
    query: str,
    chart_type: str,
    x: str,
    y: str,
    color: str | None = None,
    title: str | None = None,
    max_points: int = DEFAULT_MAX_POINTS,
) -> str:
    """Builds a static plot image from SQL over a CSV file.

    Important usage instructions for the LLM:
    1) First inspect and understand the CSV schema before plotting.
    2) Use table name 'csv_data' in SQL.
    3) Start with schema discovery using a preview query, for example:
       - SELECT * FROM csv_data LIMIT 5
    4) After understanding columns, run a chart-focused SELECT query and call this tool.
    5) Do not call tool_show_media after this tool. The UI automatically displays
       the image returned in media_content.
    """
    try:
        import duckdb
    except ModuleNotFoundError as exc:
        raise ValueError(
            "duckdb is required for tool_plot_csv. Add duckdb to the environment."
        ) from exc

    csv_path = Path(file_path).expanduser()
    if not csv_path.exists():
        raise ValueError(f"CSV file not found: {csv_path}")
    if not csv_path.is_file():
        raise ValueError(f"Path is not a file: {csv_path}")
    if csv_path.suffix.lower() != ".csv":
        raise ValueError(f"File must have .csv extension: {csv_path}")
    if max_points < 10 or max_points > MAX_ALLOWED_POINTS:
        raise ValueError(f"max_points must be between 10 and {MAX_ALLOWED_POINTS}.")

    base_query = _normalize_sql(query)
    executable_query = f"SELECT * FROM ({base_query}) AS csv_plot_query LIMIT {max_points}"

    try:
        with duckdb.connect(database=":memory:") as connection:
            escaped_path = _sql_quote(str(csv_path))
            connection.execute(
                f"CREATE VIEW {CSV_TABLE_NAME} AS "
                f"SELECT * FROM read_csv_auto('{escaped_path}', HEADER=TRUE)"
            )
            result = connection.execute(executable_query)
            rows_raw = result.fetchall()
            if result.description is None:
                columns = []
            else:
                columns = [column[0] for column in result.description]
    except Exception as exc:
        raise ValueError(f"Failed to query CSV with DuckDB: {exc}")

    if len(columns) == 0:
        raise ValueError("Query returned no columns; cannot build chart.")
    if x not in columns:
        raise ValueError(f"x field not found in query result columns: {x}")
    if y not in columns:
        raise ValueError(f"y field not found in query result columns: {y}")
    if color is not None and color not in columns:
        raise ValueError(f"color field not found in query result columns: {color}")

    points = _extract_points(rows_raw, columns, x=x, y=y, color=color)
    if len(points) == 0:
        raise ValueError("No plottable rows were produced. Check x/y values and query output.")

    image_path = _render_static_plot_image(
        points=points,
        chart_type=chart_type,
        x_label=x,
        y_label=y,
        color_label=color,
        title=title,
    )

    output = PlotCsvOutput(
        media_content=MediaContent(type="image", url=str(image_path)),
        image_file=image_path.name,
        table_name=CSV_TABLE_NAME,
        columns=columns,
        row_count=len(points),
        applied_limit=max_points,
    )
    return output.model_dump_json()
