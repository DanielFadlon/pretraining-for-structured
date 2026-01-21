"""
Base Template Module

Provides reusable base classes for prompt templates to reduce code duplication.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class BinaryLabels:
    """Labels for binary classification tasks."""
    positive: str = "Yes"
    negative: str = "No"
    positive_long: str = "Yes."
    negative_long: str = "No."


class BasePromptTemplate(ABC):
    """
    Abstract base class for prompt templates.

    Subclasses must implement:
        - context_prefix: Text before the input data
        - question: The question being asked

    Optional overrides:
        - labels: BinaryLabels instance for customizing Yes/No responses
        - text_field: Field name in example dict (default: 'text')
        - label_field: Field name for label (default: 'label')
    """

    context_prefix: str
    question: str
    labels: BinaryLabels = BinaryLabels()
    text_field: str = "text"
    label_field: str = "label"

    def create_prompt(self, context: str, answer: str = "") -> str:
        """Build the prompt with optional answer."""
        prompt = f"{self.context_prefix}{context}\n\n{self.question}"
        if answer:
            prompt += f" {answer}"
        return prompt

    def get_answer(self, example: dict[str, Any], response_type: str = "short") -> str:
        """Get the answer string based on label and response type."""
        label = example.get(self.label_field)

        if response_type == "short":
            return self.labels.positive if label == 1 else self.labels.negative
        elif response_type == "long":
            return self.labels.positive_long if label == 1 else self.labels.negative_long
        return ""

    def format(self, example: dict[str, Any], is_train: bool = False) -> str:
        """Format example into prompt (short answer during training)."""
        context = example.get(self.text_field, "")
        answer = self.get_answer(example, "short") if is_train else ""
        return self.create_prompt(context, answer)

    def format_long(self, example: dict[str, Any], is_train: bool = False) -> str:
        """Format example into prompt (long answer during training)."""
        context = example.get(self.text_field, "")
        answer = self.get_answer(example, "long") if is_train else ""
        return self.create_prompt(context, answer)

    def format_zero_shot(self, example: dict[str, Any]) -> str:
        """Format example for zero-shot inference with instruction."""
        context = example.get(self.text_field, "")
        prompt = f"{self.context_prefix}{context}\n\n"
        prompt += "Answer in one word: 'Yes' or 'No'.\n"
        prompt += self.question
        return prompt


class BinaryClassificationTemplate(BasePromptTemplate):
    """
    Convenient class for binary classification templates.

    Usage:
        template = BinaryClassificationTemplate(
            context_prefix="Given the patient condition: ",
            question="Does this patient have diabetes?",
            labels=BinaryLabels(
                positive_long="Yes, the patient has diabetes.",
                negative_long="No, the patient does not have diabetes."
            )
        )

        prompt = template.format(example, is_train=True)
    """

    def __init__(
        self,
        context_prefix: str,
        question: str,
        labels: BinaryLabels | None = None,
        text_field: str = "text",
        label_field: str = "label",
    ):
        self.context_prefix = context_prefix
        self.question = question
        self.labels = labels or BinaryLabels()
        self.text_field = text_field
        self.label_field = label_field


class NLITemplate:
    """
    Template for Natural Language Inference tasks (entailment/contradiction/neutral).

    Usage:
        template = NLITemplate()
        prompt = template.format(example, is_train=True)
    """

    LABEL_MAP = {
        "entailment": "entailment",
        "contradiction": "contradiction",
        "neutral": "neutral",
        0: "entailment",
        1: "neutral",
        2: "contradiction",
    }

    def __init__(
        self,
        premise_field: str = "sentence1",
        hypothesis_field: str = "sentence2",
        label_field: str = "gold_label",
        include_descriptions: bool = True,
    ):
        self.premise_field = premise_field
        self.hypothesis_field = hypothesis_field
        self.label_field = label_field
        self.include_descriptions = include_descriptions

    def _get_task_description(self) -> str:
        if self.include_descriptions:
            return (
                "Task: Given a premise and a hypothesis sentence, classify the relationship between them as one of the following:\n"
                " - Entailment (the hypothesis logically follows from the premise),\n"
                " - Contradiction (the hypothesis contradicts the premise), or\n"
                " - Neutral (the hypothesis is neither entailed nor contradicted by the premise).\n"
                "Your response should be only one of the three labels: entailment, contradiction, or neutral.\n\n"
            )
        return (
            "Task: Given a premise and a hypothesis sentence, classify the relationship between them as one of the following: "
            "entailment, contradiction, neutral.\n\n"
        )

    def create_prompt(self, premise: str, hypothesis: str, answer: str = "") -> str:
        """Build the NLI prompt."""
        prompt = self._get_task_description()
        prompt += f"Input:\nPremise: {premise}\nHypothesis: {hypothesis}\n\n"
        prompt += f"Answer: {answer}"
        return prompt

    def format(self, example: dict[str, Any], is_train: bool = False) -> str:
        """Format example into prompt."""
        premise = example.get(self.premise_field, "")
        hypothesis = example.get(self.hypothesis_field, "")

        answer = ""
        if is_train:
            label = example.get(self.label_field)
            answer = self.LABEL_MAP.get(label, str(label))

        return self.create_prompt(premise, hypothesis, answer)

    def format_zero_shot(self, example: dict[str, Any]) -> str:
        """Format for zero-shot inference."""
        premise = example.get(self.premise_field, "")
        hypothesis = example.get(self.hypothesis_field, "")

        prompt = self._get_task_description()
        prompt += "Answer with ONE WORD: 'entailment', 'contradiction', or 'neutral'.\n\n"
        prompt += f"Input:\nPremise: {premise}\nHypothesis: {hypothesis}\n\n"
        prompt += "Answer:"
        return prompt


# Convenience function to create formatting functions from templates
def create_format_func(template: BasePromptTemplate | NLITemplate, response_type: str = "short"):
    """
    Create a formatting function compatible with the training script.

    Args:
        template: A template instance
        response_type: "short" for Yes/No, "long" for full sentences

    Returns:
        A function(example, is_train) -> str
    """
    if response_type == "short":
        return lambda example, is_train: template.format(example, is_train)
    elif response_type == "long":
        return lambda example, is_train: template.format_long(example, is_train)
    elif response_type == "zero_shot":
        return lambda example, is_train=False: template.format_zero_shot(example)
    else:
        raise ValueError(f"Unknown response_type: {response_type}")
