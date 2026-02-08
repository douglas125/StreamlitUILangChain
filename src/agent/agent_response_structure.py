from pydantic import BaseModel
from pydantic import Field

RESPONSE_PROMPT = """
You are allowed to suggest how the next interaction with the user should be.
To do that, append a <next_interaction></next_interaction> tag at the end of your answer with this format:
<next_interaction>
<suggested_user_follow_ups>
<q>[Question suggestion 1 that user should make]</q>
<q>[Question suggestion 1 that user should make. Use 3-5 suggestions, or zero if the <presentation_mode></presentation_mode> is yes_no or free_text]</q>
</suggested_user_follow_ups>
<presentation_mode>How to present the suggestions for the user. Allowed values: free_text, yes_no, radio_box, dropdown_box, multi_select_checkbox</presentation_mode>
</next_interaction>
"""


class AgentResponseFormat(BaseModel):
    """Template for agent responses."""

    response: str = Field(description="Agent's response to the user question")
    question_suggestions: list[str] = Field(
        description="A list of suggestions of follow-up questions for the user."
    )
