"""
Sentiment Analysis Templates

Templates for sentiment classification tasks.
"""

from src.templates.base import BinaryClassificationTemplate, BinaryLabels, create_format_func


# =============================================================================
# SST-2 - Stanford Sentiment Treebank (Binary)
# =============================================================================
class SST2Template(BinaryClassificationTemplate):
    """SST-2 specific template with Positive/Negative labels."""

    def __init__(self):
        super().__init__(
            context_prefix="Consider the following sentence: '",
            question="Based on its content, determine whether the sentiment expressed is positive or negative?",
            labels=BinaryLabels(
                positive="Positive",
                negative="Negative",
                positive_long="The sentiment is positive.",
                negative_long="The sentiment is negative.",
            ),
            text_field="sentence",
            label_field="label",
        )

    def create_prompt(self, context: str, answer: str = "") -> str:
        """Build the SST-2 prompt with closing quote."""
        prompt = f"{self.context_prefix}{context}'. {self.question}"
        if answer:
            prompt += f" {answer}"
        return prompt


sst2_template = SST2Template()

sst2_one_word_prompt_formatting_func = create_format_func(sst2_template, "short")
