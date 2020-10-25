import string
import re
import nltk

def processing(doc):
    lemmatizer = nltk.stem.WordNetLemmatizer()
    stopwords = nltk.corpus.stopwords.words('english')
    english_words = set(nltk.corpus.words.words())
    pontuacao = string.punctuation
    # ETAPA 1 - Limpeza de texto
    # Converte para minúsculo
    doc = doc.lower()       

    # Trocar numeros pela string numero
    doc = re.sub(r'[0-9]+', 'numero', doc)

    # Trocar underlines por underline
    doc = re.sub(r'[_]+', 'underline', doc)

    # Trocar URL pela string httpaddr
    doc = re.sub(r'(http|https)://[^\s]*', 'httpaddr', doc)

    # Trocar Emails pela string emailaddr
    doc = re.sub(r'[^\s]+@[^\s]+', 'emailaddr', doc) 

    # Remover caracteres especiais
    doc = re.sub(r'\\r\\n', ' ', doc)
    doc = re.sub(r'\W', ' ', doc)

    # Remove caracteres simples de uma letra
    doc = re.sub(r'\s+[a-zA-Z]\s+', ' ', doc)
    doc = re.sub(r'\^[a-zA-Z]\s+', ' ', doc) 

    # Substitui multiplos espaços por um unico espaço
    doc = re.sub(r'\s+', ' ', doc, flags=re.I)

    # ETAPA 2 - Tratamento da cada palavra
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