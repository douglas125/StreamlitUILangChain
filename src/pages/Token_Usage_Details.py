import altair as alt
import streamlit as st

from src.ui.token_usage import build_invocation_usage_long_form
from src.ui.token_usage import build_invocation_metadata_rows
from src.ui.token_usage import get_thread_token_usage_invocations
from src.ui.token_usage import get_usage_metric_keys
from src.ui.token_usage import get_usage_metric_labels
from src.ui.timing_metrics import build_invocation_timing_long_form
from src.ui.timing_metrics import get_timing_metric_keys
from src.ui.timing_metrics import get_timing_metric_labels


def main():
    st.set_page_config(page_title="Token Usage Details", page_icon="Chat")
    st.title("Token Usage Details")

    ui_connector = st.session_state.get("ui_connector")
    if ui_connector is None:
        st.warning("No active chat session found.")
        st.page_link("app.py", label="Back to chat")
        return

    invocations = get_thread_token_usage_invocations(
        ui_connector.agent, ui_connector.thread_id
    )
    if not invocations:
        st.info("No token usage data found for this thread.")
        st.page_link("app.py", label="Back to chat")
        return

    metric_keys = get_usage_metric_keys()
    metric_labels = get_usage_metric_labels()
    long_data = build_invocation_usage_long_form(
        invocations, metric_keys, metric_labels
    )
    metric_order = {key: idx for idx, key in enumerate(metric_keys)}
    for row in long_data:
        row["metric_order"] = metric_order.get(row.get("metric"), 999)
    timing_keys = get_timing_metric_keys()
    timing_labels = get_timing_metric_labels()
    timing_data = build_invocation_timing_long_form(
        invocations, timing_keys, timing_labels
    )
    metadata_rows = build_invocation_metadata_rows(invocations)
    rate_data = []
    for row in invocations:
        rate = row.get("output_tokens_per_second")
        if rate is None:
            pass
        else:
            rate_data.append(
                {
                    "invocation": row.get("invocation"),
                    "tokens_per_second": rate,
                    "model_name": row.get("model_name"),
                    "model_provider": row.get("model_provider"),
                }
            )

    token_chart = (
        alt.Chart(alt.Data(values=long_data))
        .mark_bar()
        .encode(
            x=alt.X("invocation:O", title="Invocation"),
            y=alt.Y("tokens:Q", title="Tokens", stack="zero"),
            color=alt.Color(
                "metric_label:N",
                title="Metric",
                sort=[metric_labels[key] for key in metric_keys],
            ),
            order=alt.Order("metric_order:Q"),
            tooltip=[
                "invocation:O",
                "metric_label:N",
                "tokens:Q",
                "model_name:N",
                "model_provider:N",
            ],
        )
    )

    st.altair_chart(token_chart, width="stretch")
    st.caption(
        "Totals are stacked by metric. Total tokens are not shown to avoid double counting."
    )

    timing_chart = (
        alt.Chart(alt.Data(values=timing_data))
        .mark_bar()
        .encode(
            x=alt.X("invocation:O", title="Invocation"),
            xOffset=alt.XOffset("metric_label:N"),
            y=alt.Y("seconds:Q", title="Seconds"),
            color=alt.Color(
                "metric_label:N",
                title="Timing metric",
                sort=[timing_labels[key] for key in timing_keys],
            ),
            tooltip=[
                "invocation:O",
                "metric_label:N",
                "seconds:Q",
                "model_name:N",
                "model_provider:N",
            ],
        )
    )
    st.altair_chart(timing_chart, width="stretch")
    st.caption("Timing metrics are shown as grouped bars per invocation.")

    if rate_data:
        rate_chart = (
            alt.Chart(alt.Data(values=rate_data))
            .mark_bar()
            .encode(
                x=alt.X("invocation:O", title="Invocation"),
                y=alt.Y("tokens_per_second:Q", title="Output tokens / second"),
                tooltip=[
                    "invocation:O",
                    "tokens_per_second:Q",
                    "model_name:N",
                    "model_provider:N",
                ],
            )
        )
        st.altair_chart(rate_chart, width="stretch")
        st.caption("Output tokens per second uses total time minus TTFT.")
    st.markdown("**Invocation details**")
    st.dataframe(metadata_rows, hide_index=True)
    st.page_link("app.py", label="Back to chat")


if __name__ == "__main__":
    main()
