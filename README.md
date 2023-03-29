# ReadMe
## *Materials*

In the GitHub folder, the following file will be found:
1. Emosoc.py
2. original_dataset.xlsx
2. 1000_trials_emosoc_balanced.xlsx
3. arrayemosoc_balanced.npy

### EmoSoc.py
This the proper code, once run it will show the average scores of the alghoritm (accuracy and precision) after 1000 trials. 

The script is completely commented.

N.B. We have put the line that generates the excel file as a comment in order to avoid generating a new file every time the code is run, for simplicity we uploaded the excel file results of the script under the name of `1000_trials_emosoc_balanced.xlsx`.

### original_dataset.xlsx

This dataset is extracted by selecting the word that have been classified as emotional (labelled with *1*) and social (labelled with *0*) concept-related in Villani et al., (2019). Note that this file is containing an unbalanced number of items, the selection process is computed later in the python script.

### 1000_trials_emosoc_balanced.xlsx
This is an excel file containing the ending results of 1000 runs of the NBayes alghoritm.

For every single trial, each word (one for each row) can be labelled as **training** if in that specific run it was part of the training set or **match** vs. **mismatch** if it was included in the test set. Moreover, for each word, the last three columns show how many times the word was correctly classified, misclassified (when the word was in the test set).

### arrayemosoc_balanced.npy

This is an numpy array of dimension 162 x 128. It contains the word embeddings of the words present in the `original_dataset.xlsx`.
It has been computed from the twitter dataset provided by Cimino et al., (2018).

The code that generates the array is the following:


    import pandas
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy

    df = pandas.read_csv('twitter_dataset.csv')
    df2 = df.drop(['key','ranking'],axis=1)

    word2idx = {}
        for index, row in df.iterrows():
        key = row[0]
        word2idx[key] = index

    idx2word = {value: key for key, value in word2idx.items()}

    matrix = df2.to_numpy()

    df = pandas.read_excel('original_dataset.xlsx')
    words = df.iloc[:, 0]

    vector = []
    for w in words:
        vector.append(matrix[word2idx[w]])
    array = numpy.vstack(vector)
    numpy.save('arrayemosoc_balanced.npy',array)

## References



`code`
