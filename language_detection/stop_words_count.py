# %% # Language Classification
# %% ## Setting up Language and Articles
from collections import Counter
import re
import numpy as np
from nltk.corpus import stopwords
import wikipedia as wiki

stopwords_by_language = {
    'french': stopwords.words('french'),
    'english': stopwords.words('english'),
}


def get_random_wikipedia_pages(n):
    for title in wiki.random(n):
        try:
            page = wiki.page(title)
            yield page.content
        except:
            pass


wiki.set_lang("en")
english_pages = list(get_random_wikipedia_pages(50))
wiki.set_lang("fr")
french_pages = list(get_random_wikipedia_pages(50))


def count_words(text, words_to_count):
    text_tokens = re.split(r'\s+', text.lower())
    words_count = Counter(text_tokens)
    return {
        key: sum(words_count.get(word, 0) for word in words) / sum(words_count.values())
        for key, words in words_to_count.items()
    }


def vectorize(word_counts):
    return np.array([(count['french'], count['english']) for count in word_counts])


# %% ## Building Vectors

vectors = vectorize(count_words(text, stopwords_by_language) for text in english_pages + french_pages)

# %% ## Plot the Articles in the created Space
# hidden=true
from matplotlib import pyplot as plt

dot_design = {'alpha': 0.5, 'edgecolors': 'none', 's': 150}

eng = plt.scatter(*zip(*vectors[:len(english_pages)]), c='red', label='English', **dot_design)
fr = plt.scatter(*zip(*vectors[len(english_pages):]), c='blue', label='French', **dot_design)

plt.xlabel('ratio of French stop words')
plt.ylabel('ratio of English stop words')
plt.legend()
plt.show()
