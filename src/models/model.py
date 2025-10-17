from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split


class Model:
    def __init__(self, clf):
        self.clf = clf
        self.pipeline = None
        self.X = None
        self.y = None
        self.test_size = 0.3
        self.random_state = 42

        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

        self.y_pred = None

    def set_clf(self, clf):
        self.clf = clf

    def get_clf(self):
        return self.clf

    def set_X_train(self, X_train):
        self.X_train = X_train

    def get_X_train(self):
        return self.X_train

    def set_y_train(self, y_train):
        self.y_train = y_train

    def get_y_train(self):
        return self.y_train

    def set_X_test(self, X_test):
        self.X_test = X_test

    def get_X_test(self):
        return self.X_test

    def set_y_test(self, y_test):
        self.y_test = y_test

    def get_y_test(self):
        return self.y_test

    def set_X(self, X):
        self.X = X

    def get_X(self):
        return self.X

    def set_y(self, y):
        self.y = y

    def get_y(self):
        return self.y

    def set_test_size(self, test_size):
        self.test_size = test_size

    def get_test_size(self):
        return self.test_size

    def get_pipeline(self):
        return self.pipeline

    def set_random_state(self, random_state):
        self.random_state = random_state

    def get_random_state(self):
        return self.random_state

    def build_pipeline(self):
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer()),
            ("classifier", self.clf)
        ])

    def train(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, stratify=self.y, random_state=self.random_state
        )
        self.pipeline.fit(self.X_train, self.y_train)
        self.y_pred = self.pipeline.predict(self.X_test)


    def count_accuracy(self):
        return accuracy_score(self.y_test, self.y_pred), classification_report(self.y_test, self.y_pred)



