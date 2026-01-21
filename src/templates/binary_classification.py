"""
Binary Classification Templates

All binary (Yes/No) classification task templates consolidated in one place.
"""

from templates.base import BinaryClassificationTemplate, BinaryLabels, create_format_func


# =============================================================================
# BANK - Term Deposit Subscription
# =============================================================================
bank_template = BinaryClassificationTemplate(
    context_prefix="Based on the following bank information about the client:\n",
    question="Does this client subscribe to a term deposit?",
    labels=BinaryLabels(
        positive="Yes",
        negative="No",
        positive_long="Yes, the client is subscribed to a term deposit.",
        negative_long="No, the client is not subscribed to a term deposit.",
    ),
)

bank_one_word_prompt_formatting_func = create_format_func(bank_template, "short")
bank_long_res_prompt_formatting_func = create_format_func(bank_template, "long")
bank_zero_shot_prompt_formatting_func = create_format_func(bank_template, "zero_shot")

# =============================================================================
# CALHOUSING - California Housing Price Prediction
# =============================================================================
calhousing_template = BinaryClassificationTemplate(
    context_prefix=(
        "Given the features of a California census block group (1990), "
        "predict whether its median house value is above or below the statewide median.\n"
        "Features: "
    ),
    question="Is the value above the statewide median?",
    labels=BinaryLabels(
        positive="Yes",
        negative="No",
        positive_long="Yes, the district's median house value is above the statewide median.",
        negative_long="No, the district's median house value is below the statewide median.",
    ),
)

calhousing_one_word_prompt_formatting_func = create_format_func(calhousing_template, "short")
calhousing_long_response_prompt_formatting_func = create_format_func(calhousing_template, "long")


# =============================================================================
# CREDIT-G - German Credit Risk Prediction
# =============================================================================
credit_template = BinaryClassificationTemplate(
    context_prefix=(
        "Given the features of a loan applicant, "
        "predict whether the applicant is a GOOD credit risk.\n"
        "Features: "
    ),
    question="Is this applicant a GOOD credit risk?",
    labels=BinaryLabels(
        positive="Yes",
        negative="No",
        positive_long="Yes, the applicant is a GOOD credit risk.",
        negative_long="No, the applicant is a BAD credit risk.",
    ),
)

credit_one_word_prompt_formatting_func = create_format_func(credit_template, "short")
credit_long_response_prompt_formatting_func = create_format_func(credit_template, "long")

# =============================================================================
# HEART - Heart Disease Prediction
# =============================================================================
heart_template = BinaryClassificationTemplate(
    context_prefix="Given the patient condition: ",
    question="Does the coronary angiography of this patient show a heart disease?",
    labels=BinaryLabels(
        positive="Yes",
        negative="No",
        positive_long="Yes, heart disease is present.",
        negative_long="No, the patient does not have heart disease.",
    ),
)

heart_one_word_prompt_formatting_func = create_format_func(heart_template, "short")
heart_long_response_prompt_formatting_func = create_format_func(heart_template, "long")
heart_zero_shot_prompt = create_format_func(heart_template, "zero_shot")


# =============================================================================
# INCOME - Income Level Prediction (>50K)
# =============================================================================
income_template = BinaryClassificationTemplate(
    context_prefix="Based on the following information about a person:\n",
    question="Does this person earn more than 50000 dollars per year?",
    labels=BinaryLabels(
        positive="Yes",
        negative="No",
        positive_long="Yes, this person earns more than 50000 dollars per year.",
        negative_long="No, this person does not earn more than 50000 dollars per year.",
    ),
)

income_one_word_prompt_formatting_func = create_format_func(income_template, "short")
income_long_res_prompt_formatting_func = create_format_func(income_template, "long")
income_zero_shot_prompt_formatting_func = create_format_func(income_template, "zero_shot")
