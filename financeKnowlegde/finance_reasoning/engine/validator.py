FORBIDDEN_WORDS = [
    "ai",
    "model",
    "confidence",
    "accuracy",
    "prediction",
    "probability",
    "algorithm",
    "machine"
]

def validate_explanation(text):
    lower = text.lower()

    for word in FORBIDDEN_WORDS:
        if word in lower:
            raise ValueError(
                f"Forbidden language detected: {word}"
            )

    return True
