import re
import tarfile
from sklearn.feature_extraction.text import CountVectorizer


def clean_chat(text):
    cleaned_text = re.sub(r"[^a-zA-Z.,!?;:'\"()\[\]{}\-_/ ]+", "", text)
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
    return cleaned_text


def load_chat(nums_docs):
    path = "ChatGroup.tar"
    with tarfile.open(path) as tar:
        datafile = tar.extractfile('ChatGroup.txt')
        content = datafile.read().decode('utf-8', errors='ignore')
        lines = content.splitlines()[:nums_docs]
        cleaned_text = [clean_chat(line) for line in lines]
        return cleaned_text

    
def make_matrix (docs, binary=False):
    vec = CountVectorizer(min_df=5,max_df=0.9,binary=binary)
    mtx = vec.fit_transform(docs)
    cols = [None] * len(vec.vocabulary_)
    for word, idx in vec.vocabulary_.items():
        cols[idx] = word
    return mtx, cols
