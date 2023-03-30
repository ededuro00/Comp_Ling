# ReadMe
## *Materials*

In the GitHub folder, the following file will be found:
1. Emosoc.py
2. original_dataset.xlsx
3. 1000_trials_emosoc_balanced.xlsx
4. arrayemosoc_balanced.npy
5. EmoSoc_dataset.xlsx

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

### EmoSoc_daataset.xlsx

Clean dataset containing the words subdivided by cluster, with the performance of the classifier.

## References
- Cimino A., De Mattei L., Dell’Orletta F. (2018) “Multi-task Learning in Deep Neural Networks at EVALITA 2018“. In Proceedings of EVALITA ’18, Evaluation of NLP and Speech Tools for Italian, 12-13 December, Turin, Italy.

- Villani, C., Lugli, L., Liuzza, M. T., \& Borghi, A. M. (2019). Varieties of abstract concepts and their multiple dimensions. Language and Cognition: An Interdisciplinary Journal of Language and Cognitive Science, 11(3), 403–430. https://doi.org/10.1017/langcog.2019.23



