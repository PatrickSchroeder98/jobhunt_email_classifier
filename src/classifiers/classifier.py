from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB


class Classifier:
    """Class responsible for classifiers used in predictive model."""

    def __init__(self):
        """Init method that initializes the classifier as none object."""
        self.classifier = None

    def set_classifier(self, classifier):
        """Method that can set classifier."""
        self.classifier = classifier

    def get_classifier(self):
        """Method that returns classifier."""
        return self.classifier

    def set_clf_nb(self):
        """Method that can set classifier as MultinomialNB."""
        self.set_classifier(MultinomialNB())

    def set_clf_lr(self):
        """Method that can set classifier as LogisticRegression."""
        self.set_classifier(LogisticRegression())