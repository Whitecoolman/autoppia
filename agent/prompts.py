"""System and user prompt construction for the web automation agent.

Builds structured prompts that present the task, page state, interactive
elements, and action history to the LLM for decision-making.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a web automation agent. You receive a task, page state, and interactive elements. Choose ONE action and return JSON only.

Think step by step. Only return "done" when the task is clearly achieved (e.g. a confirmation message is visible, or the goal stated in the task is satisfied). Do not assume submission or navigation succeeded without evidence from the page.

Use the direct URL when the task gives one. For forms: fill required fields then click submit; do not assume submission succeeded without evidence. If the last step failed (see HISTORY), try a different element or action. Avoid repeating the same candidate_id if the previous action failed.

Actions:
- {"action":"click","candidate_id":N}
- {"action":"type","candidate_id":N,"text":"..."}
- {"action":"select","candidate_id":N,"text":"..."}
- {"action":"navigate","url":"http://localhost/..."}
- {"action":"scroll_down"} or {"action":"scroll_up"}
- {"action":"done"}

Rules:
1. candidate_id must match an [N] from INTERACTIVE ELEMENTS.
2. For navigate, keep all query params (especially ?seed=).
3. Use exact credential values from the task.
4. Return valid JSON only. No markdown or commentary.

Example:
Input: [0] button "Log In" (id=login-btn) [1] input[text] "Username" (name=user)
Task: Log in with username admin
Output: {"action":"type","candidate_id":1,"text":"admin"}
"""


# ---------------------------------------------------------------------------
# History formatting
# ---------------------------------------------------------------------------

def format_history_entry(
    step: int,
    action_type: str,
    element_text: str,
    result: str,
    url_changed: str | None = None,
) -> str:
    """Format a single history entry for the LLM prompt.

    Returns:
        ``"Step {step}: {action_type} on '{element_text}' -> {result}"``
        with optional ``" (now at {url_changed})"`` suffix.
    """
    line = f"Step {step}: {action_type} on '{element_text}' -> {result}"
    if url_changed:
        line += f" (now at {url_changed})"
    return line


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_system_prompt() -> str:
    """Return the system prompt string."""
    return SYSTEM_PROMPT


def build_user_prompt(
    *,
    task_prompt: str,
    page_ir: str,
    history_lines: list[str],
    steps_remaining: int,
    loop_hint: str | None = None,
    last_action_failed: bool = False,
) -> str:
    """Build the user message for the LLM.

    Includes task description, page IR (URL, title, page structure,
    interactive elements), action history, steps remaining with urgency,
    optional loop-detection warning, and optional last-action-failed hint.
    """
    history_text = "\n".join(history_lines) if history_lines else "No actions yet"

    # Task context: first sentence of task so the model keeps the goal in mind
    task_context = task_prompt.strip().split(".")[0].strip()
    if not task_context:
        task_context = task_prompt.strip()[:120] if task_prompt.strip() else ""

    parts = []
    if task_context:
        parts.append(f"TASK CONTEXT: {task_context}")
    parts.extend([
        f"TASK: {task_prompt}",
        "",
        page_ir,
        "",
        "HISTORY:",
        history_text,
        "",
    ])

    # Steps remaining with urgency at 3 or fewer
    steps_line = f"STEPS REMAINING: {steps_remaining}"
    if steps_remaining <= 3:
        steps_line += " -- Take the most direct action to complete the task."
    parts.append(steps_line)

    if last_action_failed:
        parts.append("")
        parts.append("Last action failed. Try a different action or element.")

    if loop_hint:
        parts.append("")
        parts.append(f"WARNING: {loop_hint}")

    parts.append("")
    parts.append("Choose your next action. Return JSON only.")

    return "\n".join(parts)
