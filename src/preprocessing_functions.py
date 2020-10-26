import string
import re
import nltk

def processing(doc):
    lemmatizer = nltk.stem.WordNetLemmatizer()
    stopwords = nltk.corpus.stopwords.words('english')
    english_words = set(nltk.corpus.words.words())
    pontuacao = string.punctuation
    # Convert to lowercase

    doc = doc.lower()       

    # change numbers with the number string
    doc = re.sub(r'[0-9]+', 'numero', doc)

    # change underlines for underline
    doc = re.sub(r'[_]+', 'underline', doc)

    # change URL for string httpaddr
    doc = re.sub(r'(http|https)://[^\s]*', 'httpaddr', doc)

    # change Emails for string emailaddr
    doc = re.sub(r'[^\s]+@[^\s]+', 'emailaddr', doc) 

    # remove special characters
    doc = re.sub(r'\\r\\n', ' ', doc)
    doc = re.sub(r'\W', ' ', doc)

    # Remove single characters from a letter
    doc = re.sub(r'\s+[a-zA-Z]\s+', ' ', doc)
    doc = re.sub(r'\^[a-zA-Z]\s+', ' ', doc) 

    # Replaces multiple spaces with a single space
    doc = re.sub(r'\s+', ' ', doc, flags=re.I)

    palavras = []
    for word in nltk.word_tokenize(doc):
        if word in stopwords:
            continue
        if word in pontuacao:
            continue
        if word not in english_words:
            continue

        word = lemmatizer.lemmatize(word)
        palavras.append(word)

    return palavras