"""
text_cleaner.py
---------------
Shared cleaning pipeline imported by both app.py and streamlit_app.py.

WARNING: Do NOT modify this file after training.
         The model learned from text cleaned exactly this way.
         Any change here breaks prediction accuracy silently.
"""

import re

CONTRACTIONS = {
    "don't":"do not", "doesn't":"does not", "didn't":"did not",
    "isn't":"is not", "aren't":"are not", "wasn't":"was not",
    "weren't":"were not", "haven't":"have not", "hasn't":"has not",
    "hadn't":"had not", "won't":"will not", "wouldn't":"would not",
    "can't":"cannot", "couldn't":"could not", "shouldn't":"should not",
    "mustn't":"must not", "needn't":"need not", "i'm":"i am",
    "i've":"i have", "i'll":"i will", "i'd":"i would",
    "it's":"it is", "it'd":"it would", "it'll":"it will",
    "that's":"that is", "they're":"they are", "they've":"they have",
    "they'll":"they will", "they'd":"they would", "we're":"we are",
    "we've":"we have", "we'll":"we will", "we'd":"we would",
    "you're":"you are", "you've":"you have", "you'll":"you will",
    "you'd":"you would", "he's":"he is", "she's":"she is",
    "there's":"there is", "who's":"who is", "what's":"what is",
    "let's":"let us", "y'all":"you all", "gonna":"going to",
    "wanna":"want to", "gotta":"got to", "ain't":"is not",
    "'ve":" have", "'re":" are", "'ll":" will", "'d":" would"
}


def clean_text(text: str) -> str:
    """
    Full cleaning pipeline in the correct order.

    Order matters:
      1. lowercase            - contraction dict lookup works
      2. expand_contractions  - BEFORE punctuation (don't -> do not)
      3. remove_urls
      4. remove_non_ascii     - strips Malayalam, Arabic, etc.
      5. remove_punctuation
      6. normalize_repeated   - reallly -> really
      7. strip whitespace
    """
    text = str(text).lower()

    for contraction, expansion in sorted(
        CONTRACTIONS.items(), key=lambda x: len(x[0]), reverse=True
    ):
        text = text.replace(contraction, expansion)

    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = text.encode('ascii', errors='ignore').decode()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
