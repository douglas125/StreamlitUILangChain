import altair as alt
import streamlit as st

from src.ui.token_usage import build_invocation_usage_long_form
from src.ui.token_usage import build_invocation_metadata_rows
from src.ui.token_usage import get_thread_token_usage_invocations
from src.ui.token_usage import get_usage_metric_keys
from src.ui.token_usage import get_usage_metric_labels


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
    metadata_rows = build_invocation_metadata_rows(invocations)

    chart = (
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
            tooltip=[
                "invocation:O",
                "metric_label:N",
                "tokens:Q",
                "model_name:N",
                "model_provider:N",
            ],
        )
    )

    st.altair_chart(chart, width="stretch")
    st.caption(
        "Totals are stacked by metric. Total tokens are not shown to avoid double counting."
    )
    st.markdown("**Invocation details**")
    st.dataframe(metadata_rows, hide_index=True)
    st.page_link("app.py", label="Back to chat")


if __name__ == "__main__":
    main()
