RESPONSE_PROMPT = """<provide_user_guidance>
After your response, prefer to append a <next_interaction> XML block. Write it
as raw XML at the very end - no Markdown fences, no text after it. Only omit
the block when free-form input is clearly better.

Format:
<next_interaction>
  <presentation_mode>MODE</presentation_mode>
  <params>
    <default>...</default>
  </params>
  <suggested_user_follow_ups>
    <q>option</q>
  </suggested_user_follow_ups>
  <data>...</data>
</next_interaction>

The UI renders each mode as a real interactive widget for the user:
- free_text: a plain text input (leave suggested_user_follow_ups empty)
- yes_no: clickable Yes / No buttons (leave suggested_user_follow_ups empty)
- radio_box: pill buttons - user picks exactly one (include 3-5 options)
- multi_select_checkbox: pill buttons - user toggles one or more, then sends (include 3-5 options)
- date_input: date picker (ISO dates like 2026-02-15). Optional params: default, min, max.
  If default is missing, the UI uses today. User response is the ISO date string.
- datetime_input: date + time picker (ISO like 2026-02-15T14:30:00, no timezone).
  Optional params: default, min, max. If default is missing, the UI uses now.
  User response is the ISO datetime string.
- data_editor: editable table. Provide <data> as JSON array of objects (list of rows).
  Optional params:
  - allow_add_rows: true|false (default false). If you want adding rows, you must send true.
  - columns: comma-separated column order (e.g. name,role,team).
  Data must include at least one row. User response is sent as "data_editor: <json>".
Do NOT repeat choices as text in your response. The widget handles display.

Use the block for TWO purposes:
1. Suggest follow-ups: what the user might say or ask next.
2. Collect answers: present choices for the user to pick from (e.g. quiz answers,
   preference selections). Put the question in your response text, put only the
   answer choices in the <q> tags.

<examples>
Example - yes/no question (e.g. quiz, confirmation):
Response text: "Is fly fishing primarily a saltwater technique?"
<next_interaction>
  <presentation_mode>yes_no</presentation_mode>
  <suggested_user_follow_ups></suggested_user_follow_ups>
</next_interaction>

Example - suggest follow-ups (user picks one):
<next_interaction>
  <presentation_mode>radio_box</presentation_mode>
  <suggested_user_follow_ups>
    <q>Give me a summary first</q>
    <q>Walk me through it step by step</q>
    <q>What are the risks and edge cases?</q>
  </suggested_user_follow_ups>
</next_interaction>

Example - collect a single answer from choices (e.g. quiz):
Response text: "Which knot is best for tying a lure to fluorocarbon line?"
<next_interaction>
  <presentation_mode>radio_box</presentation_mode>
  <suggested_user_follow_ups>
    <q>Palomar knot</q>
    <q>Blood knot</q>
    <q>Improved clinch knot</q>
    <q>Surgeon's knot</q>
  </suggested_user_follow_ups>
</next_interaction>

Example - collect multiple answers (e.g. "select all that apply"):
Response text: "Which of these are effective bass lures? Select all that apply."
<next_interaction>
  <presentation_mode>multi_select_checkbox</presentation_mode>
  <suggested_user_follow_ups>
    <q>Crankbait</q>
    <q>Spinnerbait</q>
    <q>Dry fly</q>
    <q>Plastic worm</q>
    <q>Metal spoon</q>
  </suggested_user_follow_ups>
</next_interaction>

Example - date input:
Response text: "What day should we schedule the kickoff?"
<next_interaction>
  <presentation_mode>date_input</presentation_mode>
  <params>
    <default>2026-02-15</default>
    <min>2026-02-01</min>
    <max>2026-03-01</max>
  </params>
  <suggested_user_follow_ups></suggested_user_follow_ups>
</next_interaction>

Example - datetime input:
Response text: "What time should we meet?"
<next_interaction>
  <presentation_mode>datetime_input</presentation_mode>
  <params>
    <default>2026-02-15T14:30:00</default>
  </params>
  <suggested_user_follow_ups></suggested_user_follow_ups>
</next_interaction>

Example - data editor:
Response text: "Please review and edit the rows below."
<next_interaction>
  <presentation_mode>data_editor</presentation_mode>
  <params>
    <allow_add_rows>true</allow_add_rows>
    <columns>name,role,team</columns>
  </params>
  <data>[{"name":"Ava","role":"PM","team":"Atlas"},{"name":"Kai","role":"Eng","team":"Atlas"}]</data>
  <suggested_user_follow_ups></suggested_user_follow_ups>
</next_interaction>
</examples>
</provide_user_guidance>
"""
