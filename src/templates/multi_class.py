"""
Multi-Class Classification Templates

Templates for tasks with 3+ classes: NLI, question classification, etc.
"""

from typing import Any

from src.templates.base import NLITemplate, create_format_func


# =============================================================================
# SNLI - Stanford Natural Language Inference
# =============================================================================
class SNLITemplate(NLITemplate):
    """SNLI-specific template with different label mapping and format."""

    LABEL_MAP = {
        0: "Entailment.",
        1: "Neutral.",
        2: "Contradiction.",
        "entailment": "Entailment.",
        "neutral": "Neutral.",
        "contradiction": "Contradiction.",
    }

    def create_prompt(self, premise: str, hypothesis: str, answer: str = "", is_zero_shot: bool = False) -> str:
        """Build the SNLI prompt."""
        task_instruction = " Classify the relationship as: Entailment, Neutral, or Contradiction."
        if is_zero_shot:
            task_instruction += " (Respond with exactly one label)"

        prompt = f"You are given a premise and a hypothesis.{task_instruction}\n"
        prompt += f"Premise: {premise}\n"
        prompt += f"Hypothesis: {hypothesis}\n\n"
        prompt += f"Answer: {answer}"
        return prompt

    def format(self, example: dict[str, Any], is_train: bool = False) -> str:
        """Format example into prompt."""
        premise = example.get("premise", "")
        hypothesis = example.get("hypothesis", "")

        answer = ""
        if is_train:
            label = example.get("label")
            answer = self.LABEL_MAP.get(label, "")

        return self.create_prompt(premise, hypothesis, answer)

    def format_zero_shot(self, example: dict[str, Any]) -> str:
        """Format for zero-shot inference with label emphasis."""
        premise = example.get("premise", "")
        hypothesis = example.get("hypothesis", "")
        prompt = self.create_prompt(premise, hypothesis, "", is_zero_shot=True)
        prompt = prompt[:-8] + "Answer just the Label. Answer:"
        return prompt


snli_template = SNLITemplate(
    premise_field="premise",
    hypothesis_field="hypothesis",
    label_field="label",
)

snli_one_word_prompt_formatting_func = create_format_func(snli_template, "short")
snli_inference_emphasis = lambda example: snli_template.format_zero_shot(example)

# Aliases for backward compatibility
nli_one_word_prompt_formatting_func = snli_one_word_prompt_formatting_func
nli_inference_emphasis = snli_inference_emphasis


# =============================================================================
# TREC - Question Classification (6 classes)
# =============================================================================
class TRECTemplate:
    """
    TREC question classification template.

    Classes:
        - ABBR: Abbreviation
        - DESC: Description/abstract concept
        - ENTY: Entity
        - HUM: Human
        - LOC: Location
        - NUM: Numeric value
    """

    LABEL_MAP = {
        0: "ABBR",
        1: "DESC",
        2: "ENTY",
        3: "HUM",
        4: "LOC",
        5: "NUM",
        "ABBR": "ABBR",
        "DESC": "DESC",
        "ENTY": "ENTY",
        "HUM": "HUM",
        "LOC": "LOC",
        "NUM": "NUM",
    }

    LABEL_DESCRIPTIONS = {
        "ABBR": "Abbreviation",
        "DESC": "Description or abstract concept",
        "ENTY": "Entity",
        "HUM": "Human",
        "LOC": "Location",
        "NUM": "Numeric value",
    }

    def __init__(
        self,
        text_field: str = "text",
        label_field: str = "coarse_label",
        include_descriptions: bool = True,
    ):
        self.text_field = text_field
        self.label_field = label_field
        self.include_descriptions = include_descriptions

    def _get_task_description(self) -> str:
        if self.include_descriptions:
            return (
                "Task: Classify the following question into one of these categories:\n"
                " - ABBR (Abbreviation)\n"
                " - DESC (Description or abstract concept)\n"
                " - ENTY (Entity)\n"
                " - HUM (Human)\n"
                " - LOC (Location)\n"
                " - NUM (Numeric value)\n\n"
            )
        return "Task: Classify the question as: ABBR, DESC, ENTY, HUM, LOC, or NUM.\n\n"

    def create_prompt(self, question: str, answer: str = "") -> str:
        """Build the TREC classification prompt."""
        prompt = self._get_task_description()
        prompt += f"Question: {question}\n\n"
        prompt += f"Category: {answer}"
        return prompt

    def format(self, example: dict[str, Any], is_train: bool = False) -> str:
        """Format example into prompt."""
        question = example.get(self.text_field, "")

        answer = ""
        if is_train:
            label = example.get(self.label_field)
            answer = self.LABEL_MAP.get(label, str(label))

        return self.create_prompt(question, answer)

    def format_zero_shot(self, example: dict[str, Any]) -> str:
        """Format for zero-shot inference."""
        question = example.get(self.text_field, "")

        prompt = self._get_task_description()
        prompt += "Answer with exactly ONE label: ABBR, DESC, ENTY, HUM, LOC, or NUM.\n\n"
        prompt += f"Question: {question}\n\n"
        prompt += "Category:"
        return prompt


trec_template = TRECTemplate(
    text_field="text",
    label_field="coarse_label",
    include_descriptions=True,
)

trec_template_short = TRECTemplate(
    text_field="text",
    label_field="coarse_label",
    include_descriptions=False,
)

trec_one_word_prompt_formatting_func = lambda example, is_train: trec_template.format(example, is_train)
trec_zero_shot_prompt = lambda example: trec_template.format_zero_shot(example)
