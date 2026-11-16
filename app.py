import streamlit as st
import nltk
import joblib
import pandas as pd

from nltk.sentiment import SentimentIntensityAnalyzer
from nlp_utils import clean_text, heuristic_sentiment

# Configure the Streamlit page
st.set_page_config(page_title="Airline Tweet Sentiment Analyzer")

# Download required NLTK resources safely
try:
    nltk.download("vader_lexicon", quiet=True)
    nltk.download("averaged_perceptron_tagger_eng", quiet=True)
except Exception:
    pass

# Load the ML model only once
@st.cache_resource
def load_model():
    return joblib.load("sentiment_model.joblib")

# Load VADER once
@st.cache_resource
def load_vader():
    return SentimentIntensityAnalyzer()

# Load resources
model = load_model()
vader = load_vader()

# Sidebar information
st.sidebar.title("About the Project")
st.sidebar.write("Airline Tweet Sentiment Analyzer")
st.sidebar.write("Dataset: Twitter US Airline Sentiment")
st.sidebar.write("Predicts positive, negative and neutral tweets.")

# Main title
st.title("Airline Tweet Sentiment Analyzer")
st.write("Enter a tweet to analyze its sentiment.")

# Example tweets
examples = [
    "The airline service was excellent!",
    "My flight was delayed again. Very disappointed.",
    "I have a flight tomorrow."
]

# Fill the text box when an example button is clicked
def set_example(tweet):
    st.session_state["tweet_input"] = tweet

# Display three example buttons
col1, col2, col3 = st.columns(3)

with col1:
    st.button("Positive Example", on_click=set_example, args=(examples[0],))

with col2:
    st.button("Negative Example", on_click=set_example, args=(examples[1],))

with col3:
    st.button("Neutral Example", on_click=set_example, args=(examples[2],))

# Tweet input box
tweet = st.text_area("Enter your tweet:", key="tweet_input")

# Select prediction approach
approach = st.radio(
    "Choose an approach:",
    ["Heuristic", "VADER", "ML Model"]
)

# Predict button
if st.button("Predict"):

    # Check empty input
    if not tweet.strip():
        st.warning("Please enter a tweet.")
        st.stop()

    # Limit input length
    if len(tweet) > 500:
        tweet = tweet[:500]
        st.info("Your tweet was shortened to 500 characters.")

    # Clean the tweet
    cleaned_tweet = clean_text(tweet)

    # Check cleaned input
    if not cleaned_tweet:
        st.warning("No usable text remains after cleaning.")
        st.stop()

    st.subheader("Cleaned Tweet")
    st.write(cleaned_tweet)

    # Heuristic prediction
    if approach == "Heuristic":
        prediction = heuristic_sentiment(cleaned_tweet)

    # VADER prediction
    elif approach == "VADER":
        score = vader.polarity_scores(cleaned_tweet)["compound"]

        if score >= 0.05:
            prediction = "positive"
        elif score <= -0.05:
            prediction = "negative"
        else:
            prediction = "neutral"

        st.write("VADER compound score:", score)

    # ML model prediction
    else:
        prediction = model.predict([cleaned_tweet])[0]

        # Get class probabilities
        probabilities = model.predict_proba([cleaned_tweet])[0]

        # Create probability table
        probability_df = pd.DataFrame({
            "Sentiment": model.classes_,
            "Probability": probabilities
        })

        st.subheader("Class Probabilities")
        st.bar_chart(probability_df.set_index("Sentiment"))

    # Display prediction with a coloured message
    st.subheader("Prediction")

    if prediction == "positive":
        st.success("Positive Sentiment")

    elif prediction == "negative":
        st.error("Negative Sentiment")

    else:
        st.warning("Neutral Sentiment")

    # Display POS tags
    with st.expander("View POS Tags"):
        try:
            tokens = cleaned_tweet.split()
            tagged_words = nltk.pos_tag(tokens)

            pos_table = pd.DataFrame(
                tagged_words,
                columns=["Word", "POS Tag"]
            )

            st.dataframe(pos_table, use_container_width=True)

        except LookupError:
            st.warning("NLTK POS tagging data is not available.")