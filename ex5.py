import streamlit as st
from gtts import gTTS
import os
import speech_recognition as sr
from pydub import AudioSegment
import io
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
import random

# ------------------- Fix MissingCorpusError -------------------
import textblob.download_corpora
from nltk.data import find

def ensure_corpora():
    try:
        find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', download_dir='/tmp')
        nltk.data.path.append('/tmp')
    try:
        find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger', download_dir='/tmp')
        nltk.data.path.append('/tmp')
    try:
        find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', download_dir='/tmp')
        nltk.data.path.append('/tmp')
    try:
        textblob.download_corpora.download_all()
    except Exception:
        pass

ensure_corpora()

# ------------------- Streamlit App -------------------
st.title("🧠 Unstructured Data Analysis")

tab1, tab2, tab3 = st.tabs(["🖼️ Image Analysis", "🎧 Audio Analysis", "📝 Text Analysis"])

with tab3:
    # Sample stories
    stories = [
        "In a remote kingdom nestled between jagged mountains and endless forests, Princess Elara spent her days exploring the sprawling royal gardens...",
        "During the bustling era of the 1920s, in a city that never slept, Detective Samuel Hart navigated the labyrinthine streets of New York...",
        "On a distant exoplanet, where the sky shimmered in surreal hues of emerald and violet, Captain Rhea led a team of explorers...",
        "In the neon-lit heart of Tokyo, young coder Akira toiled over lines of code that promised to revolutionize urban transportation...",
        "Deep in the Amazon rainforest, a team of scientists embarked on an unprecedented expedition to discover rare medicinal plants..."
    ]

    # Initialize session_state
    if "text_area" not in st.session_state:
        st.session_state.text_area = ""

    # Random story button
    if st.button("🎲 Random Story"):
        st.session_state.text_area = random.choice(stories)

    # Text input
    st.session_state.text_area = st.text_area(
        "Paste or modify your text here:",
        value=st.session_state.text_area,
        height=250
    )

    # Analyze button
    if st.button("Analyze Text 🚀"):
        text = st.session_state.text_area.strip()
        if text:
            blob = TextBlob(text)
            words_and_tags = blob.tags

            # POS extraction
            nouns = [word for word, tag in words_and_tags if tag.startswith('NN') and word.isalpha()]
            verbs = [word for word, tag in words_and_tags if tag.startswith('VB') and word.isalpha()]
            adjectives = [word for word, tag in words_and_tags if tag.startswith('JJ') and word.isalpha()]
            adverbs = [word for word, tag in words_and_tags if tag.startswith('RB') and word.isalpha()]

            # WordCloud function
            def make_wordcloud(words, color):
                words = [w for w in words if w.strip()]
                if not words:
                    st.warning("No words found for this category.")
                    return None
                text_for_wc = " ".join(words)
                wc = WordCloud(width=500, height=400, background_color='black', colormap=color).generate(text_for_wc)
                fig, ax = plt.subplots()
                ax.imshow(wc, interpolation='bilinear')
                ax.axis("off")
                return fig

            # Layout 2x2
            col1, col2 = st.columns(2)
            col3, col4 = st.columns(2)

            with col1:
                st.markdown("### 🧠 Nouns")
                fig = make_wordcloud(nouns, "plasma")
                if fig: st.pyplot(fig)

            with col2:
                st.markdown("### ⚡ Verbs")
                fig = make_wordcloud(verbs, "inferno")
                if fig: st.pyplot(fig)

            with col3:
                st.markdown("### 🎨 Adjectives")
                fig = make_wordcloud(adjectives, "cool")
                if fig: st.pyplot(fig)

            with col4:
                st.markdown("### 💨 Adverbs")
                fig = make_wordcloud(adverbs, "magma")
                if fig: st.pyplot(fig)

            # POS stats
            st.markdown("### 📊 POS Counts")
            st.write({
                "Nouns": len(nouns),
                "Verbs": len(verbs),
                "Adjectives": len(adjectives),
                "Adverbs": len(adverbs)
            })

            # Sentiment analysis
            st.markdown("### 💬 Sentiment Analysis")
            sentiment = blob.sentiment
            st.write(f"**Polarity:** {sentiment.polarity:.2f}")
            st.write(f"**Subjectivity:** {sentiment.subjectivity:.2f}")
        else:
            st.warning("Please paste or select some text first.")
