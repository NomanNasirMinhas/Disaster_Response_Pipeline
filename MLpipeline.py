from sqlalchemy import create_engine
import pandas as pd
import nltk
nltk.download(['punkt', 'wordnet'])
from pandas import DataFrame
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier

import numpy as np
from sklearn.metrics import classification_report
from sklearn.utils.multiclass import type_of_target
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
(type_of_target(y_test), type_of_target(y_train))

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,AdaBoostClassifier
from sklearn.metrics import make_scorer, accuracy_score, f1_score, fbeta_score, classification_report
from scipy.stats import hmean
from scipy.stats.mstats import gmean
from sklearn.externals import joblib


# load data from database
engine = create_engine('sqlite:///disaster_response.db')
df = pd.read_sql_table("messages", engine)
X = df['message']
Y = df.iloc[:,4:]
df.head()


#Data Cleaner
url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
alphanumeric_regex = '[^A-Za-z0-9]+'
def tokenize(text):
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    detected_char = re.findall(alphanumeric_regex, text)
    for chars in detected_char:
        text = text.replace(chars, ' ')
        
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    #print(clean_tokens)
    return clean_tokens


#Pipeline
pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', RandomForestClassifier())
    ])

X_train, X_test, y_train, y_test = train_test_split(X, Y)

pipeline.fit(X_train, y_train)


class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

def display_results(y_test, y_train):
    labels = np.unique(y_train)
    confusion_mat = confusion_matrix(y_test, y_train, labels=labels)
    accuracy = (y_train == y_test).mean()

    print("Labels:", labels)
    print("Confusion Matrix:\n", confusion_mat)
    print("Accuracy:", accuracy)


y_pred = pipeline.predict(X_test)

y_pred_pd = pd.DataFrame(y_pred, columns = y_test.columns)
for column in y_test.columns:
    print('------------------------------------------------------\n')
    print('FEATURE: {}\n'.format(column))
    print(classification_report(y_test[column],y_pred_pd[column]))


class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


def new_model_pipeline():
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

#             ('starting_verb', StartingVerbExtractor())
        ])),

        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])

    return pipeline

def multioutput_fscore(y_true,y_pred,beta=1):
    score_list = []
    if isinstance(y_pred, pd.DataFrame) == True:
        y_pred = y_pred.values
    if isinstance(y_true, pd.DataFrame) == True:
        y_true = y_true.values
    for column in range(0,y_true.shape[1]):
        score = fbeta_score(y_true[:,column],y_pred[:,column],beta,average='weighted')
        score_list.append(score)
    f1score_numpy = np.asarray(score_list)
    f1score_numpy = f1score_numpy[f1score_numpy<1]
    f1score = gmean(f1score_numpy)
    return  f1score



model = new_model_pipeline()
parameters = {
    'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
    'features__text_pipeline__vect__max_df': (0.75, 1.0),
    'features__text_pipeline__vect__max_features': (None, 5000),
    'features__text_pipeline__tfidf__use_idf': (True, False),
#    'clf__n_estimators': [10, 100],
#    'clf__learning_rate': [0.01, 0.1],
#    'features__transformer_weights': (
#        {'text_pipeline': 1, 'starting_verb': 0.5},
#        {'text_pipeline': 0.5, 'starting_verb': 1},
#        {'text_pipeline': 0.8, 'starting_verb': 1},
#    )
}
scorer = make_scorer(multioutput_fscore,greater_is_better = True)
cv = GridSearchCV(model, param_grid=parameters, scoring = scorer,verbose = 2, n_jobs = -1)
cv.fit(X_train, y_train)


saved = joblib.dump(cv, 'cvModel1.pkl')

load_model = joblib.load('cvModel1.pkl')
load_model.predict(X_test) 