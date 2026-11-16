import re

# Use the SAME word lists from your Task 3 notebook
positive_words = [
    "good", "great", "excellent", "happy", "love",
    "amazing", "best", "wonderful", "thank"
]

negative_words = [
    "bad", "poor", "terrible", "worst", "hate",
    "delay", "delayed", "angry", "disappointed"
]

# Clean tweet text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^a-z\s']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# Predict sentiment using the heuristic approach
def heuristic_sentiment(text):
    words = text.split()

    positive_count = 0
    negative_count = 0

    for i, word in enumerate(words):
        # Check whether the word is preceded by negation
        negated = any(
            previous_word in ["no", "not", "never", "nor"]
            for previous_word in words[max(0, i - 2):i]
        )

        if word in positive_words:
            if negated:
                negative_count += 1
            else:
                positive_count += 1

        elif word in negative_words:
            if negated:
                positive_count += 1
            else:
                negative_count += 1

    # Return the final sentiment
    if positive_count > negative_count:
        return "positive"
    elif negative_count > positive_count:
        return "negative"
    else:
        return "neutral"