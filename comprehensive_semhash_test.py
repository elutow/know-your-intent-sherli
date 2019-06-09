from time import time
import csv
import math
import random

from collections import OrderedDict
from nltk.corpus import wordnet
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, CountVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.utils.extmath import density
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import spacy

# ## Benchmarking using SemHash on NLU Evaluation Corpora
#
# This notebook benchmarks the results on the 3 NLU Evaluation Corpora:
# 1. Ask Ubuntu Corpus
# 2. Chatbot Corpus
# 3. Web Application Corpus
#
#
# More information about the dataset is available here:
#
# https://github.com/sebischair/NLU-Evaluation-Corpora
#
#
# * Semantic Hashing is used as a featurizer. The idea is taken from the paper:
#
# https://www.microsoft.com/en-us/research/publication/learning-deep-structured-semantic-models-for-web-search-using-clickthrough-data/
#
# * Benchmarks are performed on the same train and test datasets used by the other
#   benchmarks performed in the past. One important paper that benchmarks the datasets
#   mentioned above on some important platforms (Dialogflow, Luis, Watson and RASA) is :
#
# http://workshop.colips.org/wochat/@sigdial2017/documents/SIGDIAL22.pdf
#
# * Furthermore, Botfuel made another benchmarks with more platforms (Recast, Snips and their own)
#   and results can be found here:
#
# https://github.com/Botfuel/benchmark-nlp-2018
#
# * The blogposts about the benchmarks done in the past are available at :
#
# https://medium.com/botfuel/benchmarking-intent-classification-services-june-2018-eb8684a1e55f
#
# https://medium.com/snips-ai/an-introduction-to-snips-nlu-the-open-source-library-behind-snips-embedded-voice-platform-b12b1a60a41a
#
# * To be very fair on our benchmarks and results, we used the same train and test set used by the
#   other benchmarks and no cross validation or stratified splits were used. The test data was not
#   used in any way to improve the results. The dataset used can be found here:
#
# https://github.com/Botfuel/benchmark-nlp-2018/tree/master/results
#
#

# Spacy english dataset with vectors needs to be present.
# It can be downloaded using the following command:
#
# python -m spacy download en_core_web_lg
# !python -m spacy download en_core_web_lg
print('INFO: Loading spacy...')
NLP = spacy.load('en_core_web_lg')
NOUNS = {x.name().split('.', 1)[0] for x in wordnet.all_synsets('n')}
VERBS = {x.name().split('.', 1)[0] for x in wordnet.all_synsets('v')}
print('INFO: Done')

# for hyper_bench in ['AskUbuntu', 'Chatbot', 'WebApplication']:
#     benchmark_dataset = hyper

#     for hyper_over in [True, False]:
#         oversample = hyper_over

#         for hyper_syn_extra in [True, False]:
#             synonym_extra_samples = hyper_syn_extra

#             for hyper_aug in [True, False]:
#                 augm


def get_synonyms(word, number=3):
    synonyms = []
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonyms.append(l.name().lower().replace("_", " "))
    synonyms = list(OrderedDict.fromkeys(synonyms))
    return synonyms[:number]
    #return [token.text for token in most_similar(nlp.vocab[word])]


print(get_synonyms("search", -1))

#Hyperparameters
# Choose from 'AskUbuntu', 'Chatbot' or 'WebApplication'
#benchmark_dataset = ''

# Whether to oversample small classes or not. True in the paper
#oversample = False

# Whether to replace words by synonyms in the oversampled samples. True in the paper
#synonym_extra_samples = False

# Whether to add random spelling mistakes in the oversampled samples. False in the paper
#augment_extra_samples = False

# How many extra synonym augmented sentences to add for each sentence. 0 in the paper
#additional_synonyms = -1

# How many extra spelling mistake augmented sentences to add for each sentence. 0 in the paper
#additional_augments = -1

# How far away on the keyboard a mistake can be
#mistake_distance = -1

# which vectorizer to use. choose between "count", "hash", and "tfidf"
#VECTORIZER = ""

#RESULT_FILE = "result5.csv"
#METADATA_FILE = "metadata5.csv"
#NUMBER_OF_RUNS_PER_SETTING = 10
NUMBER_OF_RUNS_PER_SETTING = 1


#********* Data augmentation part **************
class MeraDataset:
    """ Class to find typos based on the keyboard distribution, for QWERTY style keyboards

        It's the actual test set as defined in the paper that we comparing against."""

    def __init__(self, dataset_path, mistake_distance, oversample,
                 augment_extra_samples, synonym_extra_samples,
                 additional_synonyms, additional_augments):
        """ Instantiate the object.
            @param: dataset_path The directory which contains the data set."""
        self.dataset_path = dataset_path
        self.X_test, self.y_test, self.X_train, self.y_train = self.load()
        self.keyboard_cartesian = {
            'q': {
                'x': 0,
                'y': 0
            },
            'w': {
                'x': 1,
                'y': 0
            },
            'e': {
                'x': 2,
                'y': 0
            },
            'r': {
                'x': 3,
                'y': 0
            },
            't': {
                'x': 4,
                'y': 0
            },
            'y': {
                'x': 5,
                'y': 0
            },
            'u': {
                'x': 6,
                'y': 0
            },
            'i': {
                'x': 7,
                'y': 0
            },
            'o': {
                'x': 8,
                'y': 0
            },
            'p': {
                'x': 9,
                'y': 0
            },
            'a': {
                'x': 0,
                'y': 1
            },
            'z': {
                'x': 0,
                'y': 2
            },
            's': {
                'x': 1,
                'y': 1
            },
            'x': {
                'x': 1,
                'y': 2
            },
            'd': {
                'x': 2,
                'y': 1
            },
            'c': {
                'x': 2,
                'y': 2
            },
            'f': {
                'x': 3,
                'y': 1
            },
            'b': {
                'x': 4,
                'y': 2
            },
            'm': {
                'x': 6,
                'y': 2
            },
            'j': {
                'x': 6,
                'y': 1
            },
            'g': {
                'x': 4,
                'y': 1
            },
            'h': {
                'x': 5,
                'y': 1
            },
            'k': {
                'x': 7,
                'y': 1
            },
            'ö': {
                'x': 11,
                'y': 0
            },
            'l': {
                'x': 8,
                'y': 1
            },
            'v': {
                'x': 3,
                'y': 2
            },
            'n': {
                'x': 5,
                'y': 2
            },
            'ß': {
                'x': 10,
                'y': 2
            },
            'ü': {
                'x': 10,
                'y': 2
            },
            'ä': {
                'x': 10,
                'y': 0
            }
        }
        self.nearest_to_i = self.get_nearest_to_i(self.keyboard_cartesian,
                                                  mistake_distance)
        self.splits = self.stratified_split(
            oversample=oversample,
            augment_extra_samples=augment_extra_samples,
            synonym_extra_samples=synonym_extra_samples,
            additional_synonyms=additional_synonyms,
            additional_augments=additional_augments)

    def get_nearest_to_i(self, keyboard_cartesian, mistake_distance):
        """ Get the nearest key to the one read.
            @params: keyboard_cartesian The layout of the QWERTY keyboard for English

            return dictionary of eaculidean distances for the characters"""
        nearest_to_i = {}
        for i in keyboard_cartesian.keys():
            nearest_to_i[i] = []
            for j in keyboard_cartesian.keys():
                if self._euclidean_distance(i,
                                            j) < mistake_distance:  #was > 1.2
                    nearest_to_i[i].append(j)
        return nearest_to_i

    def _shuffle_word(self, word, cutoff=0.7):
        """ Rearange the given characters in a word simulating typos given a probability.

            @param: word A single word coming from a sentence
            @param: cutoff The cutoff probability to make a change (default 0.9)

            return The word rearranged
            """
        word = list(word.lower())
        if random.uniform(0, 1.0) > cutoff:
            loc = np.random.randint(0, len(word))
            if word[loc] in self.keyboard_cartesian:
                word[loc] = random.choice(self.nearest_to_i[word[loc]])
        return ''.join(word)

    def _euclidean_distance(self, a, b):
        """ Calculates the euclidean between 2 points in the keyboard
            @param: a Point one
            @param: b Point two

            return The euclidean distance between the two points"""
        X = (self.keyboard_cartesian[a]['x'] -
             self.keyboard_cartesian[b]['x'])**2
        Y = (self.keyboard_cartesian[a]['y'] -
             self.keyboard_cartesian[b]['y'])**2
        return math.sqrt(X + Y)

    def _get_augment_sentence(self, sentence):
        return ' '.join(
            [self._shuffle_word(item) for item in sentence.split(' ')])

    def _augment_sentence(self, sentence, num_samples):
        """ Augment the dataset of file with a sentence shuffled
            @param: sentence The sentence from the set
            @param: num_samples The number of sentences to genererate

            return A set of augmented sentences"""
        sentences = []
        for _ in range(num_samples):
            sentences.append(self._get_augment_sentence(sentence))
        sentences = list(set(sentences))
        # print("sentences", sentences)
        return sentences + [sentence]

    def _augment_split(self, X_train, y_train, num_samples=100):
        """ Split the augmented train dataset
            @param: X_train The full array of sentences
            @param: y_train The train labels in the train dataset
            @param: num_samples the number of new sentences to create (default 1000)

            return Augmented training dataset"""
        Xs, ys = [], []
        for X, y in zip(X_train, y_train):
            tmp_x = self._augment_sentence(X, num_samples)
            for item in tmp_x:
                Xs.append(item)
                ys.append(y)
            #sample = [[Xs.append(item), ys.append(y)] for item in tmp_x]
            #print(X, y)
            #print(self.augmentedFile+str(self.nSamples)+".csv")

        with open(
                "./datasets/KL/Chatbot/train_augmented.csv", 'w',
                encoding='utf8') as csvFile:
            fileWriter = csv.writer(csvFile, delimiter='\t')
            for i in range(0, len(Xs) - 1):
                fileWriter.writerow([Xs[i] + '\t' + ys[i]])
                # print(Xs[i], "\t", ys[i])
                # print(Xs[i])
            # fileWriter.writerows(Xs + ['\t'] + ys)
        return Xs, ys

    # Randomly replaces the nouns and verbs by synonyms
    def _synonym_word(self, word, cutoff=0.5):
        if random.uniform(0, 1.0) > cutoff and len(
                get_synonyms(word)) > 0 and word in NOUNS and word in VERBS:
            return random.choice(get_synonyms(word))
        return word

    # Randomly replace words (nouns and verbs) in sentence by synonyms
    def _get_synonym_sentence(self, sentence, cutoff=0.5):
        return ' '.join(
            [self._synonym_word(item, cutoff) for item in sentence.split(' ')])

    # For all classes except the largest ones; add duplicate (possibly augmented)
    # samples until all classes have the same size
    def _oversample_split(self,
                          X_train,
                          y_train,
                          synonym_extra_samples=False,
                          augment_extra_samples=False):
        """ Split the oversampled train dataset
            @param: X_train The full array of sentences
            @param: y_train The train labels in the train dataset

            return Oversampled training dataset"""

        classes = {}
        for X, y in zip(X_train, y_train):
            if y not in classes:
                classes[y] = []
            classes[y].append(X)

        max_class_size = max([len(entries) for entries in classes.values()])

        Xs, ys = [], []
        for y in classes:
            for i in range(max_class_size):
                sentence = classes[y][i % len(classes[y])]
                if i >= len(classes[y]):
                    if synonym_extra_samples:
                        sentence = self._get_synonym_sentence(sentence)
                    if augment_extra_samples:
                        sentence = self._get_augment_sentence(sentence)
                Xs.append(sentence)
                ys.append(y)

        #with open(filename_train+"augment", 'w', encoding='utf8') as csvFile:
        #    fileWriter = csv.writer(csvFile, delimiter='\t')
        #    for i in range(0, len(Xs)-1):
        #        fileWriter.writerow([Xs[i] + '\t' + ys[i]])

        return Xs, ys

    def _synonym_split(self, X_train, y_train, additional_synonyms=100):
        """ Split the augmented train dataset
            @param: X_train The full array of sentences
            @param: y_train The train labels in the train dataset
            @param: additional_synonyms the number of new sentences to create (default 100)

            return Augmented training dataset"""
        Xs, ys = [], []
        for X, y in zip(X_train, y_train):
            for _ in range(additional_synonyms):
                Xs.append(self._get_synonym_sentence(X))
                ys.append(y)
            #sample = [[Xs.append(self._get_synonym_sentence(X)),
            #           ys.append(y)] for item in range(additional_synonyms)]
            #print(X, y)

        #with open(filename_train+"augment", 'w', encoding='utf8') as csvFile:
        #    fileWriter = csv.writer(csvFile, delimiter='\t')
        #    for i in range(0, len(Xs)-1):
        #        fileWriter.writerow([Xs[i] + '\t' + ys[i]])
        return Xs, ys

    def load(self):
        """ Load the file for now only the test.csv, train.csv files hardcoded

            return The vector separated in test, train and the labels for each one"""
        with open(self.dataset_path) as csvfile:
            readCSV = csv.reader(csvfile, delimiter='	')
            all_rows = list(readCSV)
            #             for i in all_rows:
            #                 if i ==  28823:
            #                     print(all_rows[i])
            X_test = [a[0] for a in all_rows]
            y_test = [a[1] for a in all_rows]

        with open(self.dataset_path) as csvfile:
            readCSV = csv.reader(csvfile, delimiter='\t')
            all_rows = list(readCSV)
            X_train = [a[0] for a in all_rows]
            y_train = [a[1] for a in all_rows]
        return X_test, y_test, X_train, y_train

    def process_sentence(self, x):  #pylint: disable=no-self-use
        """ Clean the tokens from stop words in a sentence.
            @param x Sentence to get rid of stop words.

            returns clean string sentence"""
        clean_tokens = []
        doc = NLP.tokenizer(x)
        for token in doc:
            if not token.is_stop:
                clean_tokens.append(token.lemma_)
        return " ".join(clean_tokens)

    def process_batch(self, X):
        """See the progress as is coming along.

            return list[] of clean sentences"""
        return [self.process_sentence(a) for a in tqdm(X)]

    def stratified_split(self, oversample, augment_extra_samples,
                         synonym_extra_samples, additional_synonyms,
                         additional_augments):
        """ Split data whole into stratified test and training sets, then remove
            stop word from sentences

            return list of dictionaries with keys train,test and values the x and y for each one"""
        self.X_train, self.X_test = ([
            preprocess(sentence) for sentence in self.X_train
        ], [preprocess(sentence) for sentence in self.X_test])
        print(self.X_train)
        if oversample:
            self.X_train, self.y_train = self._oversample_split(
                self.X_train, self.y_train, synonym_extra_samples,
                augment_extra_samples)
        if additional_synonyms > 0:
            self.X_train, self.y_train = self._synonym_split(
                self.X_train, self.y_train, additional_synonyms)
        if additional_augments > 0:
            self.X_train, self.y_train = self._augment_split(
                self.X_train, self.y_train, additional_augments)

        splits = [{
            "train": {
                "X": self.X_train,
                "y": self.y_train
            },
            "test": {
                "X": self.X_test,
                "y": self.y_test
            }
        }]
        return splits

    def get_splits(self):
        """ Get the splitted sentences

            return splitted list of dictionaries"""
        return self.splits


#****************************************************


def read_CSV_datafile(filename, intent_dict):
    X = []
    y = []
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        for row in reader:
            try:
                y.append(intent_dict[row[1]])
            except KeyError:
                # Ignore unknown intent
                print('WARN: Ignored unknown intent {}'.format(row[1]))
            else:
                X.append(row[0])
            #if benchmark_dataset == 'AskUbuntu':
            #    y.append(intent_dict[row[1]])
            #elif benchmark_dataset == 'Chatbot':
            #    y.append(intent_dict[row[1]])
            #else:
            #    y.append(intent_dict[row[1]])
    return X, y


#def tokenize(doc):
#    """
#    Returns a list of strings containing each token in `sentence`
#    """
#    tokens = []
#    doc = NLP.tokenizer(doc)
#    for token in doc:
#        tokens.append(token.text)
#    return tokens


def preprocess(doc):
    clean_tokens = []
    doc = NLP(doc)
    for token in doc:
        if not token.is_stop:
            clean_tokens.append(token.lemma_)
    return " ".join(clean_tokens)


# # SemHash


def find_ngrams(input_list, n):
    return zip(*[input_list[i:] for i in range(n)])


def semhash_tokenizer(text):
    tokens = text.split(" ")
    final_tokens = []
    for unhashed_token in tokens:
        hashed_token = "#{}#".format(unhashed_token)
        final_tokens += [
            ''.join(gram) for gram in list(find_ngrams(list(hashed_token), 3))
        ]
    return final_tokens


def semhash_corpus(corpus):
    new_corpus = []
    for sentence in corpus:
        sentence = preprocess(sentence)
        tokens = semhash_tokenizer(sentence)
        new_corpus.append(" ".join(map(str, tokens)))
    return new_corpus


def get_vectorizer(corpus, vectorizer_name):
    if vectorizer_name == "count":
        vectorizer = CountVectorizer(analyzer='word')  #,ngram_range=(1,1))
        vectorizer.fit(corpus)
        feature_names = vectorizer.get_feature_names()
    elif vectorizer_name == "hash":
        vectorizer = HashingVectorizer(
            analyzer='word', n_features=2**10, non_negative=True)
        vectorizer.fit(corpus)
        feature_names = None
    elif vectorizer_name == "tfidf":
        vectorizer = TfidfVectorizer(analyzer='word')
        vectorizer.fit(corpus)
        feature_names = vectorizer.get_feature_names()
    else:
        raise Exception(
            "{} is not a recognized Vectorizer".format(vectorizer_name))
    return vectorizer, feature_names


def trim(s):
    """Trim string to fit on terminal (assuming 80-column display)"""
    return s if len(s) <= 80 else s[:77] + "..."


# #############################################################################
# Benchmark classifiers
def benchmark(
        clf,
        X_train,
        y_train,
        X_test,
        y_test,
        target_names,
        #benchmark_dataset,
        #oversample,
        #synonym_extra_samples,
        #augment_extra_samples,
        #additional_synonyms,
        #additional_augments,
        #mistake_distance,
        print_report=True,
        feature_names=None,
        print_top10=False,
        print_cm=True):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.accuracy_score(y_test, pred)
    f1_score = metrics.f1_score(y_test, pred, average='weighted')

    #bad_pred = X_test[pred != y_test]

    print("accuracy:   %0.3f" % score)
    #print("Accuracy: %0.3f (+/- %0.3f)" % (score.mean(), score.std() * 2))

    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))

        if print_top10 and feature_names is not None:
            print("top 10 keywords per class:")
            for i, label in enumerate([
                    "Make Update", "Setup Printer", "Shutdown Computer",
                    "Software Recommendation", "None"
            ]):
                top10 = np.argsort(clf.coef_[i])[-10:]
                print(
                    trim("%s: %s" % (label, " ".join(
                        [feature_names[i] for i in top10]))))
        print()

    if print_report:
        print("classification report:")
        print(
            metrics.classification_report(
                y_test,
                pred,
                labels=range(len(target_names)),
                target_names=target_names))

    if print_cm:
        print("confusion matrix:")
        print(metrics.confusion_matrix(y_test, pred))

    #with open("./" + RESULT_FILE, 'a', encoding='utf8') as csvFile:
    #    fileWriter = csv.writer(csvFile, delimiter='\t')
    #    fileWriter.writerow([
    #        benchmark_dataset,
    #        str(clf),
    #        str(oversample),
    #        str(synonym_extra_samples),
    #        str(augment_extra_samples),
    #        str(additional_synonyms),
    #        str(additional_augments),
    #        str(mistake_distance),
    #        str(score),
    #        str(f1_score),
    #        str(train_time),
    #        str(test_time)
    #    ])

    print()
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time, f1_score


def plot_results(results):
    # make some plots
    indices = np.arange(len(results))

    results = [[x[i] for x in results] for i in range(4)]

    clf_names, score, training_time, test_time = results
    training_time = np.array(training_time) / np.max(training_time)
    test_time = np.array(test_time) / np.max(test_time)

    plt.figure(figsize=(12, 8))
    plt.title("Score")
    plt.barh(indices, score, .2, label="score", color='navy')
    plt.barh(indices + .3, training_time, .2, label="training time", color='c')
    plt.barh(
        indices + .6, test_time, .2, label="test time", color='darkorange')
    plt.yticks(())
    plt.legend(loc='best')
    plt.subplots_adjust(left=.25)
    plt.subplots_adjust(top=.95)
    plt.subplots_adjust(bottom=.05)

    for i, c in zip(indices, clf_names):
        plt.text(-.3, i, c)

    plt.show()


def data_for_training(vectorizer_name, X_train_raw, X_test_raw, y_train_raw,
                      y_test_raw):
    vectorizer, feature_names = get_vectorizer(X_train_raw, vectorizer_name)

    X_train = vectorizer.transform(X_train_raw).toarray()
    X_test = vectorizer.transform(X_test_raw).toarray()

    return X_train, y_train_raw, X_test, y_test_raw, feature_names


def evaluate_dataset(benchmark_dataset):
    #Settings from the original paper
    oversample = True
    synonym_extra_samples = True
    augment_extra_samples = False
    additional_synonyms = 0
    additional_augments = 0
    mistake_distance = 2.1
    vectorizer_name = 'tfidf'
    #for benchmark_dataset, (
    #        oversample, synonym_extra_samples, augment_extra_samples
    #), additional_synonyms, additional_augments, mistake_distance, vectorizer_name in product(
    #    ['AskUbuntu', 'Chatbot', 'WebApplication'], [(True, True, False)], [0],
    #    [0], [2.1], ["tfidf"]):

    target_names = _get_target_names(benchmark_dataset)

    intent_dict = _get_intent_dict(benchmark_dataset, target_names)

    filename_train = "datasets/KL/" + benchmark_dataset + "/train.csv"
    filename_test = "datasets/KL/" + benchmark_dataset + "/test.csv"

    print("./datasets/KL/" + benchmark_dataset + "/train.csv")
    #t0 = time()
    dataset = MeraDataset(
        dataset_path="./datasets/KL/" + benchmark_dataset + "/train.csv",
        mistake_distance=mistake_distance,
        oversample=oversample,
        augment_extra_samples=augment_extra_samples,
        synonym_extra_samples=synonym_extra_samples,
        additional_synonyms=additional_synonyms,
        additional_augments=additional_augments)

    print("mera****************************")
    splits = dataset.get_splits()
    xS_train = []
    yS_train = []

    for elem_x, elem_y in zip(splits[0]["train"]['X'],
                              splits[0]['train']['y']):
        try:
            yS_train.append(intent_dict[elem_y])
        except KeyError:
            print('WARN: Ignored unknown intent "{}"'.format(elem_y))
        else:
            xS_train.append(elem_x)
    #preprocess_time = time() - t0
    print(xS_train[:5])
    print(len(xS_train))

    X_train_raw, y_train_raw = read_CSV_datafile(
        filename=filename_train, intent_dict=intent_dict)
    X_test_raw, y_test_raw = read_CSV_datafile(
        filename=filename_test, intent_dict=intent_dict)
    print(y_train_raw[:5])
    print(X_test_raw[:5])
    print(y_test_raw[:5])
    X_train_raw = xS_train
    y_train_raw = yS_train

    print("Training data samples: \n", X_train_raw, "\n\n")

    print("Class Labels: \n", y_train_raw, "\n\n")

    print("Size of Training Data: {}".format(len(X_train_raw)))

    #
    #
    #

    #t0 = time()
    X_train_raw = semhash_corpus(X_train_raw)
    X_test_raw = semhash_corpus(X_test_raw)
    #semhash_time = time() - t0

    print(X_train_raw[:5])
    print(y_train_raw[:5])
    print()
    print(X_test_raw[:5])
    print(y_test_raw[:5])

    #t0 = time()
    X_train, y_train, X_test, y_test, feature_names = data_for_training(
        vectorizer_name=vectorizer_name,
        X_train_raw=X_train_raw,
        X_test_raw=X_test_raw,
        y_train_raw=y_train_raw,
        y_test_raw=y_test_raw)
    #vectorize_time = time() - t0

    #with open("./" + METADATA_FILE, 'a', encoding='utf8') as csvFile:
    #    fileWriter = csv.writer(csvFile, delimiter='\t')
    #    fileWriter.writerow([
    #        benchmark_dataset,
    #        str(oversample),
    #        str(synonym_extra_samples),
    #        str(augment_extra_samples),
    #        str(additional_synonyms),
    #        str(additional_augments),
    #        str(mistake_distance),
    #        str(preprocess_time),
    #        str(semhash_time),
    #        str(vectorize_time)
    #    ])

    print(X_train[0].tolist())
    print(y_train[0])
    print(feature_names)

    for _ in enumerate(range(NUMBER_OF_RUNS_PER_SETTING)):
        i_s = 0
        print("Evaluating Split {}".format(i_s))
        print("Train Size: {}\nTest Size: {}".format(X_train.shape[0],
                                                     X_test.shape[0]))
        results = []
        #alphas = np.array([1,0.1,0.01,0.001,0.0001,0])
        parameters_mlp = {
            'hidden_layer_sizes': [(100, 50), (300, 100), (300, 200, 100)]
        }
        parameters_RF = {
            "n_estimators": [50, 60, 70],
            "min_samples_leaf": [1, 11]
        }
        k_range = list(range(3, 7))
        parameters_knn = {'n_neighbors': k_range}
        knn = KNeighborsClassifier(n_neighbors=5)
        for clf, name in [
            (RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
            (GridSearchCV(knn, parameters_knn, cv=5), "gridsearchknn"),
            (GridSearchCV(
                MLPClassifier(activation='tanh'), parameters_mlp, cv=5),
             "gridsearchmlp"),
            (PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive"),
            (GridSearchCV(
                RandomForestClassifier(n_estimators=10), parameters_RF, cv=5),
             "gridsearchRF"),
            (LinearSVC(penalty='l2', dual=False, tol=1e-3), 'LinearSVC, l2'),
            (LinearSVC(penalty='l1', dual=False, tol=1e-3), 'LinearSVC, l1'),
            (SGDClassifier(alpha=.0001, n_iter=50, penalty='l2'), 'SGD, l2'),
            (SGDClassifier(alpha=.0001, n_iter=50, penalty='l1'), 'SGD, l1'),
            (SGDClassifier(alpha=.0001, n_iter=50, penalty="elasticnet"),
             'SGD with Elastic-Net penalty'),
            (NearestCentroid(),
             'NearestCentroid without threshold (aka Rocchio classifier)'),
            (MultinomialNB(alpha=.01), 'Sparse Naive Bayes (MultinomialNB)'),
            (BernoulliNB(alpha=.01), 'Sparse Naive Bayes (BernoulliNB)'),
                # The smaller C, the stronger the regularization.
                # The more regularization, the more sparsity.
            (Pipeline([('feature_selection',
                        SelectFromModel(
                            LinearSVC(penalty="l1", dual=False, tol=1e-3))),
                       ('classification', LinearSVC(penalty="l2"))]),
             'LinearSVC with L1-based feature selection'),
            (KMeans(
                n_clusters=2,
                init='k-means++',
                max_iter=300,
                verbose=0,
                random_state=0,
                tol=1e-4), 'KMeans clustering'),
            (LogisticRegression(
                C=1.0,
                class_weight=None,
                dual=False,
                fit_intercept=True,
                intercept_scaling=1,
                max_iter=100,
                multi_class='ovr',
                n_jobs=1,
                penalty='l2',
                random_state=None,
                solver='liblinear',
                tol=0.0001,
                verbose=0,
                warm_start=False), 'LogisticRegression'),
        ]:

            print('=' * 80)
            print(name)
            result = benchmark(
                clf=clf,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                target_names=target_names,
                #benchmark_dataset=benchmark_dataset,
                #oversample=oversample,
                #synonym_extra_samples=synonym_extra_samples,
                #augment_extra_samples=augment_extra_samples,
                #additional_synonyms=additional_synonyms,
                #additional_augments=additional_augments,
                #mistake_distance=mistake_distance,
                feature_names=feature_names)
            results.append(result)

        # print('parameters')
        # print(clf.grid_scores_[0])
        #print('CV Validation Score')
        # print(clf.grid_scores_[0].cv_validation_scores)
        # print('Mean Validation Score')
        # print(clf.grid_scores_[0].mean_validation_score)
        # grid_mean_scores = [result.mean_validation_score for result in clf.grid_scores_]
        # print(grid_mean_scores)
        # plt.plot(k_range, grid_mean_scores)
        # plt.xlabel('Value of K for KNN')
        # plt.ylabel('Cross-Validated Accuracy')

        #plot_results(results)

    print(len(X_train))


def _get_target_names(benchmark_dataset):
    target_names = None
    if benchmark_dataset == "Chatbot":
        target_names = ["Departure Time", "Find Connection"]
    elif benchmark_dataset == "AskUbuntu":
        target_names = [
            "Make Update", "Setup Printer", "Shutdown Computer",
            "Software Recommendation", "None"
        ]
    elif benchmark_dataset == "WebApplication":
        target_names = [
            "Download Video", "Change Password", "None", "Export Data",
            "Sync Accounts", "Filter Spam", "Find Alternative",
            "Delete Account"
        ]
    elif benchmark_dataset == 'sherli':
        with open("./datasets/KL/" + benchmark_dataset +
                  "/intents.txt") as intent_file:
            target_names = list(
                x for x in intent_file.read().splitlines() if x)
    assert target_names

    return target_names


def _get_intent_dict(benchmark_dataset, target_names):
    if benchmark_dataset == "Chatbot":
        intent_dict = {"DepartureTime": 0, "FindConnection": 1}
    elif benchmark_dataset == "AskUbuntu":
        intent_dict = {
            "Make Update": 0,
            "Setup Printer": 1,
            "Shutdown Computer": 2,
            "Software Recommendation": 3,
            "None": 4
        }
    elif benchmark_dataset == "WebApplication":
        intent_dict = {
            "Download Video": 0,
            "Change Password": 1,
            "None": 2,
            "Export Data": 3,
            "Sync Accounts": 4,
            "Filter Spam": 5,
            "Find Alternative": 6,
            "Delete Account": 7
        }
    elif benchmark_dataset == 'sherli':
        intent_dict = dict(
            (x, i) for x, i in zip(target_names, range(len(target_names))))

    return intent_dict


def train_classifiers(benchmark_dataset):
    #Settings from the original paper
    #benchmark_dataset = 'AskUbuntu'
    oversample = True
    synonym_extra_samples = True
    augment_extra_samples = False
    additional_synonyms = 0
    additional_augments = 0
    mistake_distance = 2.1
    vectorizer_name = 'tfidf'

    target_names = _get_target_names(benchmark_dataset)

    intent_dict = _get_intent_dict(benchmark_dataset, target_names)

    dataset = MeraDataset(
        dataset_path="./datasets/KL/" + benchmark_dataset + "/train.csv",
        mistake_distance=mistake_distance,
        oversample=oversample,
        augment_extra_samples=augment_extra_samples,
        synonym_extra_samples=synonym_extra_samples,
        additional_synonyms=additional_synonyms,
        additional_augments=additional_augments)

    splits = dataset.get_splits()
    xS_train = []
    yS_train = []
    for elem in splits[0]["train"]["X"]:
        xS_train.append(elem)

    for elem in splits[0]["train"]["y"]:
        try:
            yS_train.append(intent_dict[elem])
        except KeyError:
            # Ignore intents not defined in intent list
            print('WARN: Ignored unknown intent "{}"'.format(elem))

    #filename_train = "datasets/KL/" + benchmark_dataset + "/train.csv"
    #filename_test = "datasets/KL/" + benchmark_dataset + "/test.csv"
    #X_train_raw, y_train_raw = read_CSV_datafile(
    #    filename=filename_train, intent_dict=intent_dict)
    #X_test_raw, y_test_raw = read_CSV_datafile(
    #    filename=filename_test, intent_dict=intent_dict)
    #X_test_raw = [utterance]
    X_train_raw = xS_train
    y_train_raw = yS_train

    X_train_raw = semhash_corpus(X_train_raw)
    #X_test_raw = semhash_corpus(X_test_raw)

    #X_train, y_train, X_test, y_test, feature_names = data_for_training(
    #    vectorizer_name=vectorizer_name,
    #    X_train_raw=X_train_raw,
    #    X_test_raw=X_test_raw,
    #    y_train_raw=y_train_raw,
    #    y_test_raw=y_test_raw)
    y_train = y_train_raw
    vectorizer, _ = get_vectorizer(X_train_raw, vectorizer_name)

    X_train = vectorizer.transform(X_train_raw).toarray()
    #X_test = vectorizer.transform(X_test_raw).toarray()

    #results = []
    parameters_mlp = {
        'hidden_layer_sizes': [(100, 50), (300, 100), (300, 200, 100)]
    }
    parameters_RF = {"n_estimators": [50, 60, 70], "min_samples_leaf": [1, 11]}
    k_range = list(range(3, 7))
    parameters_knn = {'n_neighbors': k_range}
    knn = KNeighborsClassifier(n_neighbors=5)
    classifiers = [
        (RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
        (GridSearchCV(knn, parameters_knn, cv=5), "gridsearchknn"),
        (GridSearchCV(MLPClassifier(activation='tanh'), parameters_mlp, cv=5),
         "gridsearchmlp"),
        (PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive"),
        (GridSearchCV(
            RandomForestClassifier(n_estimators=10), parameters_RF, cv=5),
         "gridsearchRF"),
        (LinearSVC(penalty='l2', dual=False, tol=1e-3), 'LinearSVC, l2'),
        (LinearSVC(penalty='l1', dual=False, tol=1e-3), 'LinearSVC, l1'),
        (SGDClassifier(alpha=.0001, n_iter=50, penalty='l2'), 'SGD, l2'),
        (SGDClassifier(alpha=.0001, n_iter=50, penalty='l1'), 'SGD, l1'),
        (SGDClassifier(alpha=.0001, n_iter=50, penalty="elasticnet"),
         'SGD with Elastic-Net penalty'),
        (NearestCentroid(),
         'NearestCentroid without threshold (aka Rocchio classifier)'),
        (MultinomialNB(alpha=.01), 'Sparse Naive Bayes (MultinomialNB)'),
        (BernoulliNB(alpha=.01), 'Sparse Naive Bayes (BernoulliNB)'),
        # The smaller C, the stronger the regularization.
        # The more regularization, the more sparsity.
        (Pipeline([('feature_selection',
                    SelectFromModel(
                        LinearSVC(penalty="l1", dual=False, tol=1e-3))),
                   ('classification', LinearSVC(penalty="l2"))]),
         'LinearSVC with L1-based feature selection'),
        (KMeans(
            n_clusters=2,
            init='k-means++',
            max_iter=300,
            verbose=0,
            random_state=0,
            tol=1e-4), 'KMeans clustering'),
        (LogisticRegression(
            C=1.0,
            class_weight=None,
            dual=False,
            fit_intercept=True,
            intercept_scaling=1,
            max_iter=100,
            multi_class='ovr',
            n_jobs=1,
            penalty='l2',
            random_state=None,
            solver='liblinear',
            tol=0.0001,
            verbose=0,
            warm_start=False), 'LogisticRegression'),
    ]
    for clf, name in classifiers:
        print('=' * 80)
        print(name)
        clf.fit(X_train, y_train)
        #result = benchmark(
        #    clf=clf,
        #    X_train=X_train,
        #    y_train=y_train,
        #    X_test=X_test,
        #    y_test=y_test,
        #    target_names=target_names,
        #    feature_names=feature_names)
        #results.append(result)
    return classifiers, target_names, vectorizer


def predict_intent(utterance, classifiers, target_names, vectorizer):
    X_test_raw = [utterance]
    X_test_raw = semhash_corpus(X_test_raw)
    X_test = vectorizer.transform(X_test_raw).toarray()
    for clf, name in classifiers:
        print('=' * 80)
        print(name)
        result = clf.predict(X_test)
        assert len(result) == 1
        print('Intent:', target_names[result[0]])
        if hasattr(clf, 'predict_proba'):
            result_probs = clf.predict_proba(X_test)
            assert result_probs.shape[0] == 1
            for proba_name, prob in zip(target_names, result_probs[0]):
                print('{} probability: {}'.format(proba_name, prob))


def _generate_sherli_datasets():
    benchmark_dataset = 'sherli'
    with open("./datasets/KL/" + benchmark_dataset +
              "/orig_data.csv") as input_csv_file:
        #content = input_csv_file.read().replace('|', '\t').replace('\n\n', '\n')
        orig_data = tuple(
            x.split('|') for x in input_csv_file.read().splitlines() if x)
    intent_to_examples = dict()
    for example, intent in orig_data:
        intent_to_examples.setdefault(intent, list()).append(example)
    for intent in intent_to_examples:
        random.shuffle(intent_to_examples[intent])
    test_examples = list()
    train_examples = list()
    for intent in intent_to_examples:
        intent_test_size = len(intent_to_examples[intent]) // 5
        for _ in range(intent_test_size):
            test_examples.append('{}\t{}'.format(
                intent_to_examples[intent].pop(), intent))
        while intent_to_examples[intent]:
            train_examples.append('{}\t{}'.format(
                intent_to_examples[intent].pop(), intent))
    with open("./datasets/KL/" + benchmark_dataset + "/train.csv",
              'w') as output_csv_file:
        output_csv_file.write('\n'.join(train_examples))
    with open("./datasets/KL/" + benchmark_dataset + "/test.csv",
              'w') as output_csv_file:
        output_csv_file.write('\n'.join(test_examples))


def main():
    _STATE_INIT = 0
    _STATE_PREDICT = 1
    _STATE_EVALUATE = 2
    state = _STATE_INIT
    while True:
        try:
            if state == _STATE_INIT:
                user_input = input(
                    'Specify action (predict, evaluate): ').lower().strip()
                if user_input == 'evaluate':
                    state = _STATE_EVALUATE
                elif user_input == 'predict':
                    state = _STATE_PREDICT
                else:
                    print('ERROR: Invalid action. Specify again.')
                    state = _STATE_INIT
            elif state == _STATE_EVALUATE:
                benchmark_dataset = input('Dataset to evaluate: ')
                if benchmark_dataset == 'sherli':
                    _generate_sherli_datasets()
                evaluate_dataset(benchmark_dataset)
            elif state == _STATE_PREDICT:
                benchmark_dataset = input('Dataset to predict: ')
                if benchmark_dataset == 'sherli':
                    _generate_sherli_datasets()
                classifiers, target_names, vectorizer = train_classifiers(
                    benchmark_dataset)
                while True:
                    try:
                        print('#' * 80)
                        predict_intent(
                            input('Utterance: '), classifiers, target_names,
                            vectorizer)
                    except KeyboardInterrupt:
                        print()
                        break
        except KeyboardInterrupt:
            print()
            if state in (_STATE_PREDICT, _STATE_EVALUATE):
                state = _STATE_INIT
            else:
                break


if __name__ == '__main__':
    main()
