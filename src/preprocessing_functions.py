import re
from unidecode import unidecode
from nltk.data import find
from nltk import download
from nltk.corpus import stopwords
from nltk.tokenize.regexp import regexp_tokenize

# Download stopwords if it not exists
try:
    find("corpora/stopwords")
except LookupError:
    download('stopwords')

# Regex to remove stopwords
STOPWORDS = ["pra", "pras"] + stopwords.words(["english", "portuguese"])
STOPWORDS = unidecode(' '.join(STOPWORDS)).lower().split()
STOPWORDS = list(set(STOPWORDS)) # unique
STOPWORDS_RE = re.compile(r'\b(' + r'|'.join(STOPWORDS) + r')\b\s*')

# Regex to replace punctutations and numbers by single space
NON_ALPHANUM_RE = re.compile('[^a-zA-Z]')

# Regex to remove single character words
SINGLE_CHAR_RE = re.compile(r'\s+[a-zA-Z]\s+')

# Regex to replace multiple spaces by single space
MULTI_SPACE_RE = re.compile(r'\s+')

def clean_text(text, lower=True, remove_accents=True, remove_nonalpha=True, 
               remove_single_char=True, remove_stopwords=True, 
               remove_extra_space=True):
    
    if lower:
        text = text.lower()

    if remove_accents:
        text = unidecode(text)

    if remove_nonalpha:
        text = NON_ALPHANUM_RE.sub(' ', text)

    if remove_single_char:
        text = SINGLE_CHAR_RE.sub(' ', text)

    if remove_stopwords:
        text = STOPWORDS_RE.sub(' ', text)

    if remove_extra_space:
        text = MULTI_SPACE_RE.sub(' ', text)
    
    return text.strip()

def tokenize_text(text):

    words = regexp_tokenize(text, pattern="\s+", gaps=True)

    return words
