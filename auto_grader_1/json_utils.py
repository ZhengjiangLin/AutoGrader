import json
import re
from typing import Any


# Helper function: Fix common JSON errors caused by LaTeX backslashes
# (Models often output single backslash like \frac, but JSON needs \\frac)
def _repair_json_invalid_backslashes(candidate: str) -> str:
    # Valid JSON escape characters
    valid_escapes = set('"\\/bfnrtu')
    chars: list[str] = []
    in_string = False
    i = 0
    n = len(candidate)

    while i < n:
        ch = candidate[i]

        if ch == '"':
            # Check if this quote is escaped
            backslash_count = 0
            j = i - 1
            while j >= 0 and candidate[j] == "\\":
                backslash_count += 1
                j -= 1
            if backslash_count % 2 == 0:
                in_string = not in_string
            chars.append(ch)
            i += 1
            continue

        if in_string and ch == "\\":
            if i + 1 >= n:
                chars.append("\\\\")
                i += 1
                continue

            nxt = candidate[i + 1]
            if nxt in valid_escapes:
                if nxt == "u":
                    hex_part = candidate[i + 2 : i + 6]
                    if len(hex_part) == 4 and all(c in "0123456789abcdefABCDEF" for c in hex_part):
                        chars.append("\\")
                    else:
                        chars.append("\\\\")
                else:
                    chars.append("\\")
                i += 1
                continue

            # Invalid backslash → turn into double backslash
            chars.append("\\\\")
            i += 1
            continue

        chars.append(ch)
        i += 1

    return "".join(chars)


# Main function: Extract clean JSON from LLM output (which may contain markdown or extra text)
def extract_json_from_text(text: str) -> dict[str, Any]:
    """Extract valid JSON from LLM response text.

    Handles common issues:
    - Markdown code blocks (```json ... ```)
    - Extra text before/after JSON
    - Invalid backslashes in LaTeX formulas
    """
    content = text.strip()

    # Step 1: Remove markdown code block if present
    fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*\})\s*```", content)
    if fenced:
        content = fenced.group(1).strip()

    # Step 2: Try normal JSON parsing first
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Step 3: Fix backslashes and try again
        repaired = _repair_json_invalid_backslashes(content)
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass

    # Step 4: Last attempt - find the first { ... } block in the text
    brace = re.search(r"\{[\s\S]*\}", content)
    if brace:
        candidate = brace.group(0)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            repaired = _repair_json_invalid_backslashes(candidate)
            return json.loads(repaired)

    # If everything fails, raise a clear error
    raise ValueError(f"Could not extract JSON from model output: {text}")