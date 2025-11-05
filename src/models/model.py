from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split


class Model:
    """Class represents the model used in classification"""

    def __init__(self, clf):
        """Method initializes the model, classificator is given in argument."""
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
        """Method sets the classifier."""
        self.clf = clf

    def get_clf(self):
        """Method returns the classifier."""
        return self.clf

    def set_X_train(self, X_train):
        """Method sets the X training data."""
        self.X_train = X_train

    def get_X_train(self):
        """Method returns the X training data."""
        return self.X_train

    def set_y_train(self, y_train):
        """Method sets the y training data."""
        self.y_train = y_train

    def get_y_train(self):
        """Method returns the y training data."""
        return self.y_train

    def set_X_test(self, X_test):
        """Method sets the X test data."""
        self.X_test = X_test

    def get_X_test(self):
        """Method returns the X test data."""
        return self.X_test

    def set_y_test(self, y_test):
        """Method sets the y test data."""
        self.y_test = y_test

    def get_y_test(self):
        """Method returns the y test data."""
        return self.y_test

    def set_X(self, X):
        """Method sets the X data."""
        self.X = X

    def get_X(self):
        """Method returns the X data."""
        return self.X

    def set_y(self, y):
        """Method sets the y data."""
        self.y = y

    def get_y(self):
        """Method returns the y data."""
        return self.y

    def set_test_size(self, test_size):
        """Method sets the test size."""
        self.test_size = test_size

    def get_test_size(self):
        """Method returns the test size."""
        return self.test_size

    def get_pipeline(self):
        """Method returns the pipeline."""
        return self.pipeline

    def set_random_state(self, random_state):
        """Method sets the random state."""
        self.random_state = random_state

    def get_random_state(self):
        """Method returns the random state."""
        return self.random_state

    def set_y_pred(self, y_pred):
        """Method sets the y_pred."""
        self.y_pred = y_pred

    def get_y_pred(self):
        """Method returns the y_pred."""
        return self.y_pred

    def build_pipeline(self):
        """Method builds the pipeline with TfidfVectorizer and chosen classifier."""
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer()),
            ("classifier", self.clf)
        ])

    def train(self):
        """Method trains the model."""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, stratify=self.y, random_state=self.random_state
        )
        self.pipeline.fit(self.X_train, self.y_train)
        self.set_y_pred(self.pipeline.predict(self.X_test))


    def count_accuracy(self):
        """Method returns the accuracy of the model."""
        return accuracy_score(self.get_y_test(), self.get_y_pred()), classification_report(self.get_y_test(), self.get_y_pred())



