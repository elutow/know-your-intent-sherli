from collections import OrderedDict
from pathlib import Path
from time import time
from typing import List, Optional, Tuple, Any
import csv
import math
import random
import shutil
import zlib

import joblib
import numpy as np
import spacy
from nltk.corpus import wordnet
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.utils.extmath import density

# ## Benchmarking using SemHash
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

#Hyperparameters
# Whether to oversample small classes or not. True in the paper
_OVERSAMPLE = True

# Whether to replace words by synonyms in the oversampled samples. True in the paper
_SYNONYM_EXTRA_SAMPLES = True

# Whether to add random spelling mistakes in the oversampled samples. False in the paper
_AUGMENT_EXTRA_SAMPLES = False

# How many extra synonym augmented sentences to add for each sentence. 0 in the paper
_ADDITIONAL_SYNONYMS = 0

# How many extra spelling mistake augmented sentences to add for each sentence. 0 in the paper
# (NO LONGER IMPLEMENTED)
#additional_augments = -1

# How far away on the keyboard a mistake can be
_MISTAKE_DISTANCE = 2.1

# which vectorizer to use. choose between "count", "hash", and "tfidf"
_VECTORIZER_NAME = 'tfidf'

NUMBER_OF_RUNS_PER_SETTING = 1

VECTORIZER_FILENAME = 'vectorizer.joblib'
CLASSIFIER_FILENAME_SUFFIX = '_classifier.joblib'

# Runtime utilities
_DATASET_PREFIX = Path(__file__).resolve().parent / 'datasets'
_MODELS_PREFIX = Path(__file__).resolve().parent / 'models'
_NLP = None
_NOUNS = None
_VERBS = None
_INITIALIZED = False


def initialize():
    global _NLP, _NOUNS, _VERBS, _INITIALIZED  #pylint: disable=global-statement
    if _INITIALIZED:
        return
    print('INFO: Loading spacy and wordnet...')
    # Spacy english dataset with vectors needs to be present.
    # It can be downloaded using the following command:
    #
    # python -m spacy download en_core_web_lg
    # !python -m spacy download en_core_web_lg
    _NLP = spacy.load('en_core_web_lg')
    _NOUNS = {x.name().split('.', 1)[0] for x in wordnet.all_synsets('n')}
    _VERBS = {x.name().split('.', 1)[0] for x in wordnet.all_synsets('v')}
    _INITIALIZED = True
    print('INFO: Done')


def get_synonyms(word, number=3):
    synonyms = []
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonyms.append(l.name().lower().replace("_", " "))
    synonyms = list(OrderedDict.fromkeys(synonyms))
    return synonyms[:number]


#********* Data augmentation part **************
class MeraDataset:  #pylint: disable=too-many-instance-attributes
    """ Class to find typos based on the keyboard distribution, for QWERTY style keyboards

        It's the actual test set as defined in the paper that we comparing against."""

    def __init__(self, dataset_path, mistake_distance, oversample,
                 augment_extra_samples, synonym_extra_samples,
                 additional_synonyms):
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
            additional_synonyms=additional_synonyms)

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

    # Randomly replaces the nouns and verbs by synonyms
    @staticmethod
    def _synonym_word(word, cutoff=0.5):
        assert _INITIALIZED
        if random.uniform(0, 1.0) > cutoff and len(  #pylint: disable=len-as-condition
                get_synonyms(word)) > 0 and word in _NOUNS and word in _VERBS:
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

        return Xs, ys

    def load(self):
        """ Load the file for now only the test.csv, train.csv files hardcoded

            return The vector separated in test, train and the labels for each one"""
        with (self.dataset_path / 'test.csv').open() as csvfile:
            readCSV = csv.reader(csvfile, delimiter='	')
            all_rows = list(readCSV)
            X_test = [a[0] for a in all_rows]
            y_test = [a[1] for a in all_rows]

        with (self.dataset_path / 'train.csv').open() as csvfile:
            readCSV = csv.reader(csvfile, delimiter='\t')
            all_rows = list(readCSV)
            X_train = [a[0] for a in all_rows]
            y_train = [a[1] for a in all_rows]
        return X_test, y_test, X_train, y_train

    def stratified_split(self, oversample, augment_extra_samples,
                         synonym_extra_samples, additional_synonyms):
        """ Split data whole into stratified test and training sets, then remove
            stop word from sentences

            return list of dictionaries with keys train,test and values the x and y for each one"""
        self.X_train, self.X_test = ([
            preprocess(sentence) for sentence in self.X_train
        ], [preprocess(sentence) for sentence in self.X_test])
        if oversample:
            self.X_train, self.y_train = self._oversample_split(
                self.X_train, self.y_train, synonym_extra_samples,
                augment_extra_samples)
        if additional_synonyms > 0:
            self.X_train, self.y_train = self._synonym_split(
                self.X_train, self.y_train, additional_synonyms)

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


def read_CSV_datafile(filepath, intent_dict):
    X = []
    y = []
    unknown_intents = set()
    with filepath.open() as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        for row in reader:
            try:
                y.append(intent_dict[row[1]])
            except KeyError:
                # Ignore unknown intent
                if row[1] not in unknown_intents:
                    unknown_intents.add(row[1])
                    print('WARN: Ignored unknown intent {}'.format(row[1]))
            else:
                X.append(row[0])
    return X, y


def preprocess(doc):
    clean_tokens = []
    doc = _NLP(doc)
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


def _new_classifiers() -> List[Tuple[Any, str]]:
    parameters_mlp = {
        'hidden_layer_sizes': [(100, 50), (300, 100), (300, 200, 100)]
    }
    parameters_RF = {"n_estimators": [50, 60, 70], "min_samples_leaf": [1, 11]}
    k_range = list(range(3, 7))
    parameters_knn = {'n_neighbors': k_range}
    knn = KNeighborsClassifier(n_neighbors=5)
    return [
        (GridSearchCV(knn, parameters_knn, cv=5), "gridsearchknn"),
        (GridSearchCV(MLPClassifier(activation='tanh'), parameters_mlp, cv=5),
         "gridsearchmlp"),
        (GridSearchCV(
            RandomForestClassifier(n_estimators=10), parameters_RF, cv=5),
         "gridsearchRF"),
        (MultinomialNB(alpha=.01), 'Sparse_Naive_Bayes_MultinomialNB'),
        (BernoulliNB(alpha=.01), 'Sparse_Naive_Bayes_BernoulliNB'),
        # The smaller C, the stronger the regularization.
        # The more regularization, the more sparsity.
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


# #############################################################################
# Benchmark classifiers
def benchmark(  #pylint: disable=too-many-locals
        clf,
        X_train,
        y_train,
        X_test,
        y_test,
        target_names,
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

    print("accuracy:   %0.3f" % score)

    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))

        if print_top10 and feature_names is not None:
            print("top 10 keywords per class:")
            for i, label in enumerate(target_names):
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

    print()
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time, f1_score


def data_for_training(vectorizer_name, X_train_raw, X_test_raw, y_train_raw,
                      y_test_raw):
    vectorizer, feature_names = get_vectorizer(X_train_raw, vectorizer_name)

    X_train = vectorizer.transform(X_train_raw).toarray()
    X_test = vectorizer.transform(X_test_raw).toarray()

    return X_train, y_train_raw, X_test, y_test_raw, feature_names


def evaluate_dataset(benchmark_dataset):  #pylint: disable=too-many-locals
    target_names = _get_target_names(benchmark_dataset)

    intent_dict = _get_intent_dict(target_names)

    filepath_train = _DATASET_PREFIX / benchmark_dataset / 'train.csv'
    filepath_test = _DATASET_PREFIX / benchmark_dataset / 'test.csv'

    #t0 = time()
    dataset = MeraDataset(
        dataset_path=_DATASET_PREFIX / benchmark_dataset,
        mistake_distance=_MISTAKE_DISTANCE,
        oversample=_OVERSAMPLE,
        augment_extra_samples=_AUGMENT_EXTRA_SAMPLES,
        synonym_extra_samples=_SYNONYM_EXTRA_SAMPLES,
        additional_synonyms=_ADDITIONAL_SYNONYMS)

    print("mera****************************")
    splits = dataset.get_splits()
    xS_train = []
    yS_train = []

    unknown_intents = set()
    for elem_x, elem_y in zip(splits[0]["train"]['X'],
                              splits[0]['train']['y']):
        try:
            yS_train.append(intent_dict[elem_y])
        except KeyError:
            if elem_y not in unknown_intents:
                unknown_intents.add(elem_y)
                print('WARN: Ignored unknown intent "{}"'.format(elem_y))
        else:
            xS_train.append(elem_x)

    X_train_raw, y_train_raw = read_CSV_datafile(
        filepath=filepath_train, intent_dict=intent_dict)
    X_test_raw, y_test_raw = read_CSV_datafile(
        filepath=filepath_test, intent_dict=intent_dict)
    X_train_raw = xS_train
    y_train_raw = yS_train

    print("Training data samples: \n", X_train_raw, "\n\n")

    print("Class Labels: \n", y_train_raw, "\n\n")

    print("Size of Training Data: {}".format(len(X_train_raw)))

    X_train_raw = semhash_corpus(X_train_raw)
    X_test_raw = semhash_corpus(X_test_raw)

    X_train, y_train, X_test, y_test, feature_names = data_for_training(
        vectorizer_name=_VECTORIZER_NAME,
        X_train_raw=X_train_raw,
        X_test_raw=X_test_raw,
        y_train_raw=y_train_raw,
        y_test_raw=y_test_raw)

    for _ in enumerate(range(NUMBER_OF_RUNS_PER_SETTING)):
        i_s = 0
        print("Evaluating Split {}".format(i_s))
        print("Train Size: {}\nTest Size: {}".format(X_train.shape[0],
                                                     X_test.shape[0]))
        results = []
        for clf, name in _new_classifiers():

            print('=' * 80)
            print(name)
            result = benchmark(
                clf=clf,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                target_names=target_names,
                feature_names=feature_names)
            results.append(result)


def _get_target_names(benchmark_dataset, prefix=_DATASET_PREFIX) -> List[str]:
    with (prefix / benchmark_dataset / 'intents.txt').open() as intent_file:
        return list(x for x in intent_file.read().splitlines() if x)


def _get_intent_dict(target_names):
    intent_dict = dict(
        (x, i) for x, i in zip(target_names, range(len(target_names))))

    return intent_dict


def train_classifiers(benchmark_dataset):  #pylint: disable=too-many-locals
    intent_dict = _get_intent_dict(_get_target_names(benchmark_dataset))

    dataset = MeraDataset(
        dataset_path=_DATASET_PREFIX / benchmark_dataset,
        mistake_distance=_MISTAKE_DISTANCE,
        oversample=_OVERSAMPLE,
        augment_extra_samples=_AUGMENT_EXTRA_SAMPLES,
        synonym_extra_samples=_SYNONYM_EXTRA_SAMPLES,
        additional_synonyms=_ADDITIONAL_SYNONYMS)

    splits = dataset.get_splits()
    xS_train = []
    yS_train = []
    unknown_intents = set()
    for elem_x, elem_y in zip(splits[0]["train"]['X'],
                              splits[0]['train']['y']):
        try:
            yS_train.append(intent_dict[elem_y])
        except KeyError:
            if elem_y not in unknown_intents:
                unknown_intents.add(elem_y)
                print('WARN: Ignored unknown intent "{}"'.format(elem_y))
        else:
            xS_train.append(elem_x)

    X_train_raw = xS_train
    y_train_raw = yS_train

    X_train_raw = semhash_corpus(X_train_raw)
    y_train = y_train_raw
    vectorizer, _ = get_vectorizer(X_train_raw, _VECTORIZER_NAME)

    X_train = vectorizer.transform(X_train_raw).toarray()

    classifiers = _new_classifiers()
    for clf, name in classifiers:
        print('=' * 80)
        print(name)
        clf.fit(X_train, y_train)
    return classifiers, vectorizer


def get_vectorized_utterance(utterance: str, vectorizer):
    X_test_raw = semhash_corpus([utterance])
    X_test = vectorizer.transform(X_test_raw).toarray()
    return X_test


def predict_intent(vectorized_utterance, clf, target_names):
    result = clf.predict(vectorized_utterance)
    assert len(result) == 1
    result_probs = clf.predict_proba(vectorized_utterance)
    assert result_probs.shape[0] == 1
    return target_names[result[0]], dict(zip(target_names, result_probs[0]))


def print_intent_prediction(utterance, classifiers, vectorizer, target_names):
    X_test = get_vectorized_utterance(utterance, vectorizer)
    for clf, name in classifiers:
        print('=' * 80)
        print(name)
        intent, intent_probs = predict_intent(X_test, clf, target_names)
        print('Intent:', intent)
        for proba_name, prob in intent_probs.items():
            print('{} probability: {}'.format(proba_name, prob))


def _get_data_hash(orig_csv: Path, intents_txt: Path) -> int:
    return zlib.adler32(intents_txt.read_bytes(),
                        zlib.adler32(orig_csv.read_bytes(), 0))


def _do_fingerprints_match(fingerprint_path: Path, new_hash: int) -> bool:
    if fingerprint_path.is_file():
        try:
            fingerprint_hash = int(fingerprint_path.read_text().strip())
        except ValueError:
            return False
        return new_hash == fingerprint_hash
    return False


def _update_fingerprint(fingerprint_parent: Path, orig_data_path: Path,
                        intents_txt_path: Path) -> bool:
    new_hash = _get_data_hash(orig_data_path, intents_txt_path)
    fingerprint_path = fingerprint_parent / 'data.fingerprint'
    if _do_fingerprints_match(fingerprint_path, new_hash):
        return False
    fingerprint_path.write_text(str(new_hash))
    return True


def save_models(dataset_name: str, classifiers, vectorizer) -> None:
    output_dir = _MODELS_PREFIX / dataset_name
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    # Write intents.txt
    dataset_intents_txt = _DATASET_PREFIX / dataset_name / 'intents.txt'
    model_intents_txt = output_dir / 'intents.txt'
    model_intents_txt.write_bytes(dataset_intents_txt.read_bytes())

    # Write vectorizer
    joblib.dump(vectorizer, str(output_dir / VECTORIZER_FILENAME))

    # Write classifiers
    for clf, name in classifiers:
        joblib.dump(clf, str(output_dir / (name + CLASSIFIER_FILENAME_SUFFIX)))


def load_models(dataset_name: str, models_dir: Optional[Path] = None
                ) -> Tuple[List[Tuple[Any, str]], Any, List[str]]:
    input_dir: Path
    if models_dir:
        input_dir = models_dir
    else:
        input_dir = _MODELS_PREFIX / dataset_name

    # Read vectorizer
    vectorizer = joblib.load(str(input_dir / VECTORIZER_FILENAME))

    # Read classifiers
    classifiers: List[Tuple[Any, str]] = list()
    for clf_path in input_dir.glob('*' + CLASSIFIER_FILENAME_SUFFIX):
        name = clf_path.name[:-len(CLASSIFIER_FILENAME_SUFFIX)]
        clf = joblib.load(str(clf_path))
        classifiers.append((clf, name))

    # Read intent names (target_names)
    target_names = _get_target_names(dataset_name, prefix=_MODELS_PREFIX)

    return classifiers, vectorizer, target_names


def write_dataset_traintest(benchmark_dataset: str) -> None:
    orig_data_path = _DATASET_PREFIX / benchmark_dataset / 'orig_data.csv'
    intents_txt_path = _DATASET_PREFIX / benchmark_dataset / 'intents.txt'
    if not _update_fingerprint(_DATASET_PREFIX / benchmark_dataset,
                               orig_data_path, intents_txt_path):
        return
    with orig_data_path.open() as input_csv_file:
        orig_data = tuple(
            x.split('|') for x in input_csv_file.read().splitlines() if x)
    intent_to_examples = dict()
    for entry in orig_data:
        try:
            example, intent = entry
        except ValueError as exc:
            print(f'ERROR: Example is malformed: {entry}')
            raise exc
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
    with (_DATASET_PREFIX / benchmark_dataset /
          'train.csv').open('w') as output_csv_file:
        output_csv_file.write('\n'.join(train_examples))
    with (_DATASET_PREFIX / benchmark_dataset /
          'test.csv').open('w') as output_csv_file:
        output_csv_file.write('\n'.join(test_examples))


def _handle_state_predict() -> None:
    dataset_name = input('Models to load: ')
    classifiers, vectorizer, target_names = load_models(dataset_name)
    while True:
        try:
            print('#' * 80)
            print_intent_prediction(
                input('Utterance: '), classifiers, vectorizer, target_names)
        except KeyboardInterrupt:
            print()
            break


def main() -> None:  #pylint: disable=too-many-branches
    _STATE_INIT = 0
    _STATE_PREDICT = 1
    _STATE_EVALUATE = 2
    _STATE_TRAIN = 3
    state = _STATE_INIT

    # Program initialization
    initialize()

    # Main UI loop
    while True:
        try:
            if state == _STATE_INIT:
                user_input = input(
                    'Specify action (predict, evaluate, train): ').lower(
                    ).strip()
                if user_input == 'evaluate':
                    state = _STATE_EVALUATE
                elif user_input == 'predict':
                    state = _STATE_PREDICT
                elif user_input == 'train':
                    state = _STATE_TRAIN
                else:
                    print('ERROR: Invalid action. Specify again.')
                    state = _STATE_INIT
            elif state == _STATE_EVALUATE:
                benchmark_dataset = input('Dataset to evaluate: ')
                write_dataset_traintest(benchmark_dataset)
                evaluate_dataset(benchmark_dataset)
            elif state == _STATE_PREDICT:
                _handle_state_predict()
            elif state == _STATE_TRAIN:
                dataset_name = input('Dataset to train against: ')
                write_dataset_traintest(dataset_name)
                print('INFO: Training against dataset...')
                classifiers, vectorizer = train_classifiers(dataset_name)
                print('INFO: Saving models under models/{} ...'.format(
                    dataset_name))
                save_models(dataset_name, classifiers, vectorizer)
                state = _STATE_INIT
            else:
                print('ERROR: Invalid state', repr(state))
                exit(1)
        except KeyboardInterrupt:
            print()
            if state in (_STATE_PREDICT, _STATE_EVALUATE):
                state = _STATE_INIT
            else:
                break


if __name__ == '__main__':
    main()
