'''
This script would return a word2vec plot from a given wiki link

To use:
python wiki_word2vec [topic you want to search on]
'''


def get_browser(path, headless=True):
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    '''
    Uses Selenium and a pre-installed selenium chrome driver.
    Returns a haedless chrome browser  of window size 19200x1080 
    controlled by selenium
    '''
    chrome_options = Options()
    if headless == True:
        chrome_options.add_argument("--headless")
    chrome_options.add_argument("--window-size=1920x1080")
    chrome_driver = path
    browser = webdriver.Chrome(
        chrome_options=chrome_options, executable_path=chrome_driver)
    return browser


def clean_text(wiki_url):
    '''
    script that scraps a given wiki url and clean the text

    Return tokenize sentences
    '''
    import string
    import nltk
    import requests
    from bs4 import BeautifulSoup
    import re
    import html
    from nltk.corpus import stopwords

    resp = requests.get(wiki_url, timeout=5)
    soup = BeautifulSoup(resp.text, 'lxml')

    # getting all the p tags
    text = ''
    for para in soup.find_all('p'):
        text += html.unescape(para.text)
        text += para.text

    # cleaning using regex
    def clean(text):
        clean_text = html.unescape(text)
        clean_text = re.sub(r'<a.*</a>', ' ', clean_text)
        clean_text = re.sub(r'<p.*</p>', ' ', clean_text)
        clean_text = re.sub(r'<.?>', ' ', clean_text)
        clean_text = re.sub(r'\s+', ' ', clean_text)

        def decontracted(phrase):
            # specific
            phrase = re.sub(r"won't", "will not", phrase)
            phrase = re.sub(r"can\'t", "can not", phrase)

            # general
            phrase = re.sub(r"n\'t", " not", phrase)
            phrase = re.sub(r"\'re", " are", phrase)
            phrase = re.sub(r"\'s", " is", phrase)
            phrase = re.sub(r"\'d", " would", phrase)
            phrase = re.sub(r"\'ll", " will", phrase)
            phrase = re.sub(r"\'t", " not", phrase)
            phrase = re.sub(r"\'ve", " have", phrase)
            phrase = re.sub(r"\'m", " am", phrase)
            return phrase
        clean_text = decontracted(clean_text)
        return clean_text
    clean_text = clean(text)

    sentences = nltk.sent_tokenize(clean_text)

    def clean_2(text):
        # remove punctuations
        regex = re.compile('[%s]' % re.escape(string.punctuation))
        clean_text = regex.sub('', text)
        return clean_text

    sentences = [nltk.word_tokenize(clean_2(sentence))
                 for sentence in sentences]

    def clean_3(tokens):
        clean_tokens = [
            token for token in tokens if token not in stopwords.words('english')]
        pos = nltk.pos_tag(clean_tokens, tagset='universal')
        wnl = nltk.WordNetLemmatizer()
        new_tokens = []
        accepted_pos = ['NOUN', 'VERB', 'ADJ', 'ADV']
        to_lemmatize = ['NOUN', 'VERB']
        change_dict = {'NOUN': 'n',
                       'VERB': 'v',
                       'ADJ': 'a',
                       'ADV': 'r'}
        for i in pos:
            if i[-1] in accepted_pos:
                temp = i[0]
                if i[-1] in to_lemmatize:
                    temp = wnl.lemmatize(temp, pos=change_dict[i[-1]])
                temp.lower()
                new_tokens.append(temp.lower())
        return new_tokens

    sentences = [clean_3(tokens) for tokens in sentences]
    print("Number of sentences found: {}".format(len(sentences)))

    return sentences


def get_word2vec(clean_text, size=100, window=20, max_words=100, seed=123):
    import multiprocessing
    from gensim.models import Word2Vec
    cpu_count = multiprocessing.cpu_count()
    print("Getting word2vec")
    print("Number of CPUs used: {}".format(cpu_count))

    # initialize word2vec
    # getting the right amount of words
    words_count = 1e10
    min_count = 0
    while words_count > max_words:
        # print("min count: {}".format(min_count))
        model = Word2Vec(clean_text, size=size, window=window,
                         min_count=min_count, workers=cpu_count, seed=seed)
        words_count = len(model.wv.vocab)
        # print("number of words: {}".format(words_count))
        min_count += 1

    print("Size of vocab in word2vec model: {}".format(words_count))

    return model


def plot_word2vec(model, perplexity=20, n_components=2, init='pca', n_iter=2500, random_state=123, save=False):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    labels = []
    tokens = []
    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)

    tsne_model = TSNE(perplexity=perplexity, n_components=n_components,
                      init=init, n_iter=n_iter, random_state=random_state)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
    plt.figure(figsize=(16, 16))
    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.title('word2vec plot on ' + TOPIC)
    plt.axis('off')
    if save:
        plt.savefig('word2vec.png')
    plt.show()


if __name__ == '__main__':
    import time
    import sys
    import os
    from selenium.webdriver.common.keys import Keys

    # get selenium browser to perform wiki search on topic
    path = os.path.join(os.getcwd(), 'chromedriver')
    browser = get_browser(path, False)
    browser.get('https://en.wikipedia.org/wiki/Main_Page')
    search_bar = browser.find_element_by_id('searchInput')
    TOPIC = ''
    for i in sys.argv[1:]:
        TOPIC += i + ' '
    search_bar.send_keys(TOPIC)
    search_bar.send_keys(Keys.ENTER)
    time.sleep(1)  # sleep to let browser load
    url = browser.current_url
    print("url: {}".format(url))
    browser.close()

    sentences = clean_text(url)
    plot_word2vec(get_word2vec(sentences))
