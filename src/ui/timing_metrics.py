import streamlit as st


def record_thread_timing(thread_id, ttft_seconds, total_seconds):
    """Store timing metrics for a thread in Streamlit session state."""
    timings_by_thread = st.session_state.setdefault("_thread_timings", {})
    if not isinstance(timings_by_thread, dict):
        timings_by_thread = {}
        st.session_state["_thread_timings"] = timings_by_thread
    thread_timings = timings_by_thread.setdefault(thread_id, [])
    if not isinstance(thread_timings, list):
        thread_timings = []
        timings_by_thread[thread_id] = thread_timings
    invocation = len(thread_timings) + 1
    thread_timings.append(
        {
            "invocation": invocation,
            "ttft_seconds": ttft_seconds,
            "total_seconds": total_seconds,
        }
    )


def get_thread_time_metrics(thread_id):
    """Return timing metrics for a thread."""
    timings_by_thread = st.session_state.get("_thread_timings", {})
    if not isinstance(timings_by_thread, dict):
        return []
    thread_timings = timings_by_thread.get(thread_id, [])
    if not isinstance(thread_timings, list):
        return []
    return thread_timings


def merge_invocation_timings(rows, timings):
    timings_by_inv = {}
    for timing in timings:
        if isinstance(timing, dict):
            invocation = timing.get("invocation")
            if invocation is not None:
                timings_by_inv[invocation] = timing
    for row in rows:
        invocation = row.get("invocation")
        timing = timings_by_inv.get(invocation)
        if timing:
            row["ttft_seconds"] = timing.get("ttft_seconds")
            row["total_seconds"] = timing.get("total_seconds")
    return rows


def append_missing_timing_rows(rows, timings):
    existing = set()
    for row in rows:
        invocation = row.get("invocation")
        if invocation is not None:
            existing.add(invocation)
    for timing in timings:
        if not isinstance(timing, dict):
            pass
        else:
            invocation = timing.get("invocation")
            if invocation in existing:
                pass
            else:
                row = {
                    "invocation": invocation,
                    "model_name": None,
                    "model_provider": None,
                    "ttft_seconds": timing.get("ttft_seconds"),
                    "total_seconds": timing.get("total_seconds"),
                }
                rows.append(row)
                existing.add(invocation)


def build_rows_from_timings(timings, zero_row_factory):
    rows = []
    for timing in timings:
        if not isinstance(timing, dict):
            pass
        else:
            row = zero_row_factory()
            row["invocation"] = timing.get("invocation")
            row["model_name"] = None
            row["model_provider"] = None
            row["ttft_seconds"] = timing.get("ttft_seconds")
            row["total_seconds"] = timing.get("total_seconds")
            rows.append(row)
    return rows


def get_timing_metric_labels():
    return {
        "ttft_seconds": "TTFT (seconds)",
        "total_seconds": "Total time (seconds)",
    }


def get_timing_metric_keys():
    return ["ttft_seconds", "total_seconds"]


def build_invocation_timing_long_form(invocations, metrics, metric_labels):
    data = []
    for row in invocations:
        for metric in metrics:
            label = metric_labels.get(metric, metric)
            value = row.get(metric)
            if value is None:
                seconds = 0
            else:
                seconds = value
            data.append(
                {
                    "invocation": row.get("invocation"),
                    "metric": metric,
                    "metric_label": label,
                    "seconds": seconds,
                    "model_name": row.get("model_name"),
                    "model_provider": row.get("model_provider"),
                }
            )
    return data
