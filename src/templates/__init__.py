"""
Prompt Templates Module

Provides reusable prompt templates for various classification tasks.

Usage:
    # Import base classes
    from templates import BinaryClassificationTemplate, BinaryLabels

    # Import specific templates
    from templates.binary_classification import bank_template, heart_template
    from templates.multi_class import snli_template, trec_template
    from templates.sentiment import sst2_template

    # Import formatting functions
    from templates.binary_classification import bank_one_word_prompt_formatting_func
"""

from templates.base import (
    BasePromptTemplate,
    BinaryClassificationTemplate,
    BinaryLabels,
    NLITemplate,
    create_format_func,
)

# Import all templates for convenience
from templates.binary_classification import (
    bank_template,
    blood_template,
    calhousing_template,
    credit_template,
    heart_template,
    income_template,
)
from templates.multi_class import snli_template, trec_template
from templates.sentiment import sst2_template

__all__ = [
    # Base classes
    "BasePromptTemplate",
    "BinaryClassificationTemplate",
    "BinaryLabels",
    "NLITemplate",
    "create_format_func",
    # Templates
    "bank_template",
    "calhousing_template",
    "credit_template",
    "heart_template",
    "income_template",
    "snli_template",
    "trec_template",
    "sst2_template",
]
