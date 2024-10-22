import re
import string
from unicodedata import normalize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def clean_text(text):
    """
    Cleans the input text by performing several transformations such as:
    - Removing HEX characters
    - Normalizing the text (removing accents and special characters)
    - Removing extra whitespace and newline characters
    - Removing punctuation
    - Lowercasing the text
    - Tokenizing and removing stopwords
    - Returning the cleaned text as a string of words
    """

    punctuation = string.punctuation
    punctuation = punctuation.replace('+', '').replace('-', '').replace('*', '').replace('.', '').replace(',', '')
    punctuation = punctuation + '“”«»°'

    # Example usage:
    # cleaned_text = clean_text("Este es un ejemplo... de texto! Contiene (paréntesis) y --- otros símbolos.")
    # expected_output = 'ejemplo texto contiene paréntesis -- - símbolos'

    # Step 1: Define the set of stopwords in Spanish
    stop_w = set(stopwords.words('spanish'))

    # Step 2: Normalize the text to remove accents and special characters
    # Normalize text using 'NFKD' form which separates characters from their diacritics
    _clean_text = normalize('NFKD', text)

    # Step 3: Remove unnecessary whitespace (e.g., two or more spaces)
    _clean_text = re.sub(r'\s{2,}', ' ', _clean_text)

    # Step 4: Remove spaces before and after newline characters
    _clean_text = re.sub(r'\n\s+', '\n', _clean_text)  # Spaces after newline
    _clean_text = re.sub(r'\s+\n', '\n', _clean_text)  # Spaces before newline

    # Step 5: Limit consecutive newlines to a maximum of two
    _clean_text = re.sub(r'\n{3,}', '\n\n', _clean_text)

    # Step 6: Replace groups of consecutive periods with a single period
    _clean_text = re.sub(r'\.\s*\.+', '.', _clean_text)

    # Step 7: Convert all text to lowercase for uniformity
    _clean_text = _clean_text.lower()

    # Step 8: Remove all punctuation characters
    _clean_text = _clean_text.translate(str.maketrans('', '', punctuation))

    # Step 9: Tokenize the text into words (split text into individual tokens)
    _clean_text = word_tokenize(_clean_text)

    # Step 10: Remove stopwords from the tokenized words
    # List comprehension filters out any words that are in the stopwords set
    # _clean_text = [word for word in _clean_text if word not in stop_w]

    # Step 11: Join the remaining words back into a single string and return
    _clean_text = ' '.join(_clean_text)
    return _clean_text