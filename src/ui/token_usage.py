def get_thread_token_usage(agent, thread_id):
    """Aggregate token usage from checkpointed messages for a thread."""
    state = agent.get_state({"configurable": {"thread_id": thread_id}})
    messages = state.values.get("messages", [])
    return aggregate_token_usage(messages)


def get_thread_token_usage_invocations(agent, thread_id):
    """Return per-invocation token usage rows for a thread."""
    state = agent.get_state({"configurable": {"thread_id": thread_id}})
    messages = state.values.get("messages", [])
    return extract_invocation_usage(messages)


def aggregate_token_usage(messages):
    totals = zero_token_usage()
    seen = {key: False for key in totals}
    for msg in messages:
        usage = extract_usage_from_message(msg)
        if usage is None:
            pass
        else:
            for key, value in usage.items():
                if value is None:
                    pass
                else:
                    totals[key] += value
                    seen[key] = True
    return totals, seen


def extract_invocation_usage(messages):
    rows = []
    invocation = 0
    for msg in messages:
        usage = extract_usage_from_message(msg)
        if usage is None:
            pass
        else:
            invocation += 1
            row = normalize_usage_values(usage)
            row["invocation"] = invocation
            rows.append(row)
    return rows


def extract_usage_from_message(msg):
    usage_metadata = getattr(msg, "usage_metadata", None)
    if isinstance(usage_metadata, dict) and usage_metadata:
        return usage_from_metadata(usage_metadata)

    response_metadata = getattr(msg, "response_metadata", None)
    if isinstance(response_metadata, dict) and response_metadata:
        usage = response_metadata.get("usage")
        if isinstance(usage, dict) and usage:
            return usage_from_response_usage(usage)
        return usage_from_ollama_metadata(response_metadata)

    return None


def usage_from_metadata(usage_metadata):
    usage = empty_usage_values()
    usage["input_tokens"] = coerce_int(usage_metadata.get("input_tokens"))
    usage["output_tokens"] = coerce_int(usage_metadata.get("output_tokens"))
    usage["total_tokens"] = coerce_int(usage_metadata.get("total_tokens"))

    details = usage_metadata.get("input_token_details") or {}
    if isinstance(details, dict):
        usage["cache_read_input_tokens"] = coerce_int(details.get("cache_read"))
        usage["cache_creation_input_tokens"] = coerce_int(details.get("cache_creation"))
        usage["ephemeral_5m_input_tokens"] = coerce_int(
            details.get("ephemeral_5m_input_tokens")
        )
        usage["ephemeral_1h_input_tokens"] = coerce_int(
            details.get("ephemeral_1h_input_tokens")
        )

    usage = fill_total_tokens(usage)
    return usage


def usage_from_response_usage(usage):
    result = empty_usage_values()
    result["input_tokens"] = coerce_int(usage.get("input_tokens"))
    result["output_tokens"] = coerce_int(usage.get("output_tokens"))
    result["total_tokens"] = coerce_int(usage.get("total_tokens"))
    result["cache_read_input_tokens"] = coerce_int(
        usage.get("cache_read_input_tokens")
    )
    result["cache_creation_input_tokens"] = coerce_int(
        usage.get("cache_creation_input_tokens")
    )

    cache_creation = usage.get("cache_creation") or {}
    if isinstance(cache_creation, dict):
        result["ephemeral_5m_input_tokens"] = coerce_int(
            cache_creation.get("ephemeral_5m_input_tokens")
        )
        result["ephemeral_1h_input_tokens"] = coerce_int(
            cache_creation.get("ephemeral_1h_input_tokens")
        )

    result = fill_total_tokens(result)
    return result


def usage_from_ollama_metadata(response_metadata):
    usage = empty_usage_values()
    usage["input_tokens"] = coerce_int(response_metadata.get("prompt_eval_count"))
    usage["output_tokens"] = coerce_int(response_metadata.get("eval_count"))
    usage = fill_total_tokens(usage)
    if usage["input_tokens"] is None and usage["output_tokens"] is None:
        return None
    return usage


def fill_total_tokens(usage):
    total_tokens = usage.get("total_tokens")
    if total_tokens is None:
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        if input_tokens is None or output_tokens is None:
            pass
        else:
            usage["total_tokens"] = input_tokens + output_tokens
    return usage


def normalize_usage_values(usage):
    normalized = {}
    for key in zero_token_usage().keys():
        value = usage.get(key)
        normalized[key] = 0 if value is None else value
    return normalized


def empty_usage_values():
    return {
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
        "cache_read_input_tokens": None,
        "cache_creation_input_tokens": None,
        "ephemeral_5m_input_tokens": None,
        "ephemeral_1h_input_tokens": None,
    }


def zero_token_usage():
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "ephemeral_5m_input_tokens": 0,
        "ephemeral_1h_input_tokens": 0,
    }


def coerce_int(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def format_usage_table(data):
    header = "| Metric | Value |\n| --- | --- |\n"
    rows = []
    for row in data:
        metric = row.get("Metric", "")
        value = row.get("Value", "")
        rows.append(f"| {metric} | {value} |")
    return header + "\n".join(rows)


def build_invocation_usage_long_form(invocations, metrics, metric_labels):
    data = []
    for row in invocations:
        for metric in metrics:
            label = metric_labels.get(metric, metric)
            data.append(
                {
                    "invocation": row.get("invocation"),
                    "metric": metric,
                    "metric_label": label,
                    "tokens": row.get(metric, 0),
                }
            )
    return data


def get_usage_metric_labels():
    return {
        "input_tokens": "Input tokens",
        "output_tokens": "Output tokens",
        "cache_read_input_tokens": "Cache read input tokens",
        "cache_creation_input_tokens": "Cache creation input tokens",
        "ephemeral_5m_input_tokens": "Ephemeral 5m input tokens",
        "ephemeral_1h_input_tokens": "Ephemeral 1h input tokens",
    }


def get_usage_metric_keys():
    return [
        "input_tokens",
        "output_tokens",
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
        "ephemeral_5m_input_tokens",
        "ephemeral_1h_input_tokens",
    ]
