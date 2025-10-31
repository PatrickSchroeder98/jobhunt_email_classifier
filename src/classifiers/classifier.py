from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn.naive_bayes import MultinomialNB, ComplementNB, BernoulliNB
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    HistGradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier,
)
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier


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

    def set_clf_cnb(self):
        """Method that can set classifier as ComplementNB."""
        self.set_classifier(ComplementNB())

    def set_clf_bnb(self):
        """Method that can set classifier as BernoulliNB."""
        self.set_classifier(BernoulliNB())

    def set_clf_lr(self):
        """Method that can set classifier as LogisticRegression."""
        self.set_classifier(LogisticRegression())

    def set_clf_sgd(self):
        """Method that can set classifier as SGDClassifier."""
        self.set_classifier(SGDClassifier())

    def set_clf_rdg(self):
        """Method that can set classifier as RidgeClassifier."""
        self.set_classifier(RidgeClassifier())

    def set_clf_rfc(self):
        """Method that can set classifier as RandomForestClassifier."""
        self.set_classifier(RandomForestClassifier())

    def set_clf_gbc(self):
        """Method that can set classifier as GradientBoostingClassifier."""
        self.set_classifier(GradientBoostingClassifier())

    def set_clf_abc(self):
        """Method that can set classifier as AdaBoostClassifier."""
        self.set_classifier(AdaBoostClassifier())

    def set_clf_lsv(self):
        """Method that can set classifier as LinearSVC."""
        self.set_classifier(LinearSVC())

    def set_clf_svc(self):
        """Method that can set classifier as SVC."""
        self.set_classifier(SVC())

    def set_clf_knn(self):
        """Method that can set classifier as KNeighborsClassifier."""
        self.set_classifier(KNeighborsClassifier())

    def set_clf_vtc(self):
        """Method that can set classifier as VotingClassifier."""
        self.set_classifier(
            VotingClassifier(
                estimators=[
                    ("lr", LogisticRegression(max_iter=200)),
                    ("nb", MultinomialNB()),
                    ("svm", LinearSVC()),
                ],
                voting="hard",
            )
        )
