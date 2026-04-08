from typing import Any

from .config import Settings
from .json_utils import extract_json_from_text
from .llm_client import LLMClient


class Grader:
    """Handles the actual grading logic by calling the LLM with a structured prompt."""

    def __init__(self, settings: Settings, llm: LLMClient):
        self.settings = settings
        self.llm = llm

    def grade_answer(
        self,
        student_text: str,
        standard_answer: str,
        *,
        total_questions: int | None = None,
    ) -> dict[str, Any]:
        """Grade a student's answer using the LLM and return structured JSON result."""

        # Prepare hint about total number of questions (helps LLM distribute scores reasonably)
        total_questions_hint = (
            f"【题目总数】\n本次作业共 {total_questions} 题，请据此在各题间分配扣分，保证总分合理。\n"
            if total_questions
            else ""
        )

        # The prompt sent to the LLM
        prompt = f"""
You are a strict but fair teaching assistant. Grade the student's answer based on the standard answer and deduction rules.

【Standard Answer】
{standard_answer}

{total_questions_hint}

【Deduction Rules】
{self.settings.deduction_rules}

【Grading Constraints】
1) Full score is 100 points. Normal scores should be in the 85-100 range.
2) Only deduct significantly (down to 70) for major omissions, missing key steps, or obvious errors.
3) Correct answer = 0 deduction.
4) Wrong answer but good steps/ideas = 1-2 deduction per question.
5) Wrong answer with no valid steps = 3-4 deduction per question.
6) Output only total score and per-question data. Leave overall_comment as empty string by default.

【Student Answer】
{student_text}

Please output ONLY valid JSON, no Markdown:
{{
  "total_score": 0,
  "items": [{{"question_no":"1","score":0,"max_score":0,"standard_answer":"","deduction_reason":"","comment":""}}],
  "overall_comment": ""
}}
""".strip()

        # Call the LLM
        result = self.llm.chat(
            model=self.settings.grading_model,
            messages=[
                {"role": "system", "content": "You are an automatic grading assistant that only outputs JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=3000,
            response_format={"type": "json_object"},
        )

        # Extract clean JSON from the LLM's response
        return extract_json_from_text(self.llm.message_text(result))