# Text Mining on Hacker News Corpus
This repo contains the scripts to do simple text mining and analysis on the public HackerNews data set. 

# Data
Hacker News is a social news website ran by investment fund and startup incubator, Y combinator. It focuses on computer science and entrepreneurship.  <br>

## BigQuery
The complete dataset can be queried using Google Cloud Platform (GCP) from [here](https://cloud.google.com/bigquery/public-data/hacker-news). <br>

## About the dataset
This dataset contains all stories and comments from Hacker News from its launch in 2006. Each story contains a story id, the author that made the post, when it was written, and the number of points the story received.


# Setup
## Requirements
To setup the project, you would need to have [Anaconda3](https://www.anaconda.com/download/) installed. 

## Conda Env
1. Clone the repo
```
git clone https://github.com/notha99y/TextMiningHackerNews.git
```
2. Set up conda environment
```
cd TextMiningHackerNews
conda env create -f=environment.yml
```

# Business Questions

# Techniques
## Text Cleaning
### html encoding
Most of the text in the comments contain `html encoding` which we need to clean, either by dropping them or converting them to their intended display character. <br>

Fortunately, `Python3` has an `html` package which a method called `unescape` that quickly converts the `html encoding` to the human readable text.

## Topic Modelling

## Latent Dirichlet Allocation (LDA)


## Word2Vec


# Personal Notes
## Pandas

`pd.read_table`
```python
# read general delimited file into DataFrame
pd.read_table('your_corpous.txt', header = None, names = ['your', 'columns', 'names'])
```
`df.groupby('class').describe()`
```python
# aggregation
```

`df.apply(your_function)`, `df.assign(col_name = Series)`
```pytohn
length = df['class'].apply(len) # alias
df = df.assign(Length = length) # creating a new column in the dataframe
```

## NLTK
`stopwords`
```python
from ntlk.corpus import stopwords
stopwords.words('English') #  show all the stops words in the corpus
# note: need to run ntlk.download() get books
```

`Tokenizer`
```python
import nltk
# words tokenizing
tokens = nltk.word_tokenize(text)

# sentence tokenizing
sentences = nltk.sent_tokenize(text)
```

`Lemmatizer`
```python
WNlemma = nlt.WordNetLemmatizer()

WNlemma.lemmatize('better', pos = 'a')
```

`df.apply()`, `tokenizer`, `stopwords`
```python
def pre_process(text):
    tokens = nltk.word_tokenize(text)
    tokens=[WNlemma.lemmatize(t) for t in tokens]
    tokens=[word for word in tokens if word not in stopwords.words('english')]
    text_after_process=" ".join(tokens)
    return(text_after_process)
```

`POS Tagging`
```python
import nltk
from nltk import word_tokenize
from nltk import pos_tag

# Step 1 tokenize the text
tokens = word_tokenize(text)

# Step 2 apply pos_tagging
pos_1 = pos_tag(tokens)
pos_2 = pos_tag(tokens, tagset = 'universal')
pos_3 = pos_tag(tokens, tagset = 'wsj')
pos_4 = pos_tag(tokens, tagset = 'brown')
```
## Sk-learn
`train-test-split`
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.X, df.Y, test_size = 0.2, random_state = 1)
```

### Document Term Matrix
A mathematical matrix that describes the frequency of terms that occur in a collection of document (corpus). <br>

`CountVectorizer()` produces a sparse representation of the counts using
`scipy.sparse.csr_matrix`
```python
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
# 1. fit
x_train_counts = count_vect.fit_transform(x_train)
# 2. get_feature_names()
x_train_fnames = count_vect.get_feature_names()
# 3. From Dataframe
dtm = pd.DataFrame(x_train_counts.toarray(), columns = x_train_fnames)
```
### Term Frequency-Inverse Document Frequency
a numerical statistic that is intended to reflect how important a word is to a documents in a corpus. <br>

#### Usage
It is often used as a weighting factor in searches of information retrieval, text mining, and user modeling. 

#### Some notes using sklearn TfidfTransformer
if use_idf is set to `True`: Enable inverse-document-frequency reweighting.
```python
from sklearn.feature_extraction.text import TfidfTransformer
# 1. fit
tf_transformer = TfidfTransformer(use_idf=False).fit(x_train_count)
# 2. transform dataset
x_train_tf = tf_transformer.transform(x_train_count)
```
### Pipelining (SUPER USEFUL)
builds a pipeline to combine multiple steps into one
```python
from sklearn.pipeline import Pipeline
# 1. generate pipeline
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB())
])
# 2. fit
text_clf.fit(X_train, y_train)
#Test model accuracy
import numpy as np
from sklearn import metrics 
predicted = text_clf.predict(X_test)
print(metrics.confusion_matrix(y_test, predicted))
print(np.mean(predicted == y_test) )
```
