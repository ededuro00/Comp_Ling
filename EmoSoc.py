# import packages
import pandas
import random
import pandas
import os
import numpy
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import statistics
from sklearn.metrics import f1_score

# set random seed to 0
random.seed(0)

# import excel file where the original dataset is stored (227 items, 81 + 146) onto a dataframe variable
df = pandas.read_excel('original_dataset.xlsx')
words = df.iloc[:, 0]
labels = df.iloc[:, 1]

# pre-process words (putting them in lowercase)
words = [n.lower() for n in words]
words = pandas.Series(words)

# extract a random sample on indexes that will be used to drop 65
idx_to_delete = random.sample(range(0, 145), 64)

# drop words and labels at the idx_to_delete index and reset the indexes
words = words.drop(idx_to_delete)
words = words.reset_index(drop=True)
labels = labels.drop(idx_to_delete)
labels = labels.reset_index(drop=True)

# load array containing words embeddings and create idx2word and word2idx dictionaries
features = numpy.load('arrayemosoc_balanced.npy')

idx = 0
idx2word = {}
for w in words:
    idx2word[idx] = w
    idx += 1
word2idx = {v: k for k, v in idx2word.items()}

# declare accuracy and f-1 score lists (where each element will be the score for the single run)
accuracy = []
f1 = []

# initiate a dataframe that will be the 1000_trials_emosoc_balances.xlsx
result = pandas.DataFrame({'words': words})

# not consider the random seed parameter anymore
del random.seed

# for loop that iterate 1000 times the NBayes classifier
for iteration in range(1, 1001):
    # split the features, labels and words into train and test set.
    # N.B. Each run splits differently since the random_state parameter changes accordingly with the iteration variable
    features_train, features_test, labels_train, labels_test, word_train, word_test = train_test_split(features, labels,
                                                                                                       words,
                                                                                                       test_size=0.2,
                                                                                                       random_state=iteration)

    # instantiate the GaussianNB as gnb
    gnb = GaussianNB()

    # fit the alghoritm using the features and labels of the training set
    model = gnb.fit(features_train, labels_train)

    # predict the labels from the test feature set
    labels_pred = gnb.predict(features_test)

    # append the accuracy and f-1 score to the two lists previously created
    accuracy.append(accuracy_score(labels_test, labels_pred))
    f1.append(f1_score(labels_test, labels_pred, average='weighted'))

    # change type to Series object
    labels_pred_series = pandas.Series(labels_pred, index=labels_test.index)

    # create excel file
    result[f'trial {iteration}'] = 'training'

    for i in labels_test.index:
        if labels_test[i] == labels_pred_series[i]:
            result.at[i, f'trial {iteration}'] = 'match'
        else:
            result.at[i, f'trial {iteration}'] = 'mismatch'

# add the last 2 columns, n_matches and n_mismatches
result['n_matches'] = (result == 'match').sum(axis=1)
result['n_mismatches'] = (result == 'mismatch').sum(axis=1)

    #result.to_excel('1000trials_emosoc_balanced.xlsx')

# print measures
print('The mean accuracy after 1000 trials is = ', statistics.mean(accuracy),'\n')
print('The mean f1-score after 1000 trials is = ', statistics.mean(f1))

sys.exit('\nThanks for using our program, have a good day ;)\n\nEdoardo\nSofia')