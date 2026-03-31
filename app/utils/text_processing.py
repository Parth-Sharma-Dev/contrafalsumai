import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
nltk_stop_words = set(stopwords.words("english"))

important_words = {
    "no",
    "not",
    "nor",
    "never",
    "none",
    "nothing",
    "nowhere",
    "neither",
    "without",
    "against",
    "but",
    "however",
}
custom_stop_words = nltk_stop_words - important_words


def get_wordnet_pos(word: str) -> str:
    if word.endswith("ing") or word.endswith("ed"):
        return wordnet.VERB
    return wordnet.NOUN


def preprocess_text(text: str) -> str:
    if not text:
        return ""

    text = text.lower()
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"http[s]?://\S+|www\.\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = word_tokenize(text)

    processed_tokens: list[str] = []
    for word in tokens:
        if word not in custom_stop_words:
            lemmatized_word = lemmatizer.lemmatize(word, get_wordnet_pos(word))
            processed_tokens.append(lemmatized_word)

    return " ".join(processed_tokens)
