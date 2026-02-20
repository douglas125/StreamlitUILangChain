from __future__ import annotations

import datetime as dt
import decimal
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from pydantic import Field

from langchain.tools import tool

DEFAULT_ROW_LIMIT = 200
CSV_TABLE_NAME = "csv_data"
FORBIDDEN_SQL_PATTERN = re.compile(
    r"\b("
    r"insert|update|delete|create|alter|drop|truncate|copy|attach|detach|"
    r"call|pragma|vacuum|replace|merge|grant|revoke"
    r")\b",
    re.IGNORECASE,
)


class QueryCsvInput(BaseModel):
    file_path: str = Field(
        description="Local path to a CSV file that will be exposed as table 'csv_data'."
    )
    query: str = Field(
        description=(
            "SQL query to execute against table 'csv_data'. "
            "Use SELECT-style queries."
        )
    )


class QueryCsvOutput(BaseModel):
    table_name: str = Field(description="Table name available in SQL.")
    columns: list[str] = Field(description="Column names returned by the query.")
    row_count: int = Field(description="Number of rows returned.")
    rows: list[dict[str, Any]] = Field(description="Query results as row objects.")
    applied_limit: int | None = Field(
        description=(
            "Default row limit applied by the tool when the query has no explicit LIMIT."
        ),
        default=None,
    )


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


def _has_limit_clause(query: str) -> bool:
    return re.search(r"\blimit\b", query, flags=re.IGNORECASE) is not None


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
        return str(value)
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


@tool(args_schema=QueryCsvInput)
def tool_query_csv(file_path: str, query: str) -> str:
    """Runs SQL over a CSV file and returns JSON rows.

    Important usage instructions for the LLM:
    1) First inspect and understand the CSV schema before analytical queries.
    2) Use table name 'csv_data' in SQL.
    3) Start with schema discovery using a preview query, for example:
       - SELECT * FROM csv_data LIMIT 5
    4) After understanding the columns, run targeted SELECT queries.
    """
    try:
        import duckdb
    except ModuleNotFoundError as exc:
        raise ValueError(
            "duckdb is required for tool_query_csv. Add duckdb to the environment."
        ) from exc

    csv_path = Path(file_path).expanduser()
    if not csv_path.exists():
        raise ValueError(f"CSV file not found: {csv_path}")
    if not csv_path.is_file():
        raise ValueError(f"Path is not a file: {csv_path}")
    if csv_path.suffix.lower() != ".csv":
        raise ValueError(f"File must have .csv extension: {csv_path}")

    base_query = _normalize_sql(query)
    has_limit = _has_limit_clause(base_query)
    applied_limit = None
    executable_query = base_query
    if not has_limit:
        executable_query = (
            f"SELECT * FROM ({base_query}) AS csv_query_result LIMIT {DEFAULT_ROW_LIMIT}"
        )
        applied_limit = DEFAULT_ROW_LIMIT

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

    rows = []
    for row in rows_raw:
        parsed_row = {}
        for column_name, value in zip(columns, row):
            parsed_row[column_name] = _to_json_safe(value)
        rows.append(parsed_row)

    output = QueryCsvOutput(
        table_name=CSV_TABLE_NAME,
        columns=columns,
        row_count=len(rows),
        rows=rows,
        applied_limit=applied_limit,
    )
    return output.model_dump_json()
