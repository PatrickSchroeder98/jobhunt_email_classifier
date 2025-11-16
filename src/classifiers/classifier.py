from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn.naive_bayes import MultinomialNB, ComplementNB, BernoulliNB
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
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
        self.max_iter = 200
        self.vc_clf_1 = LogisticRegression(max_iter=self.max_iter)
        self.vc_clf_2 = MultinomialNB()
        self.vc_clf_3 = LinearSVC()
        self.voting = "hard"

        self.sc_clf_1 = MultinomialNB()
        self.sc_clf_2 = LinearSVC()
        self.sc_clf_3 = RandomForestClassifier()

        self.ESTIMATORS_AND_CLASSIFIERS = {
            "MultinomialNB": MultinomialNB,
            "LogisticRegression": LogisticRegression,
            "ComplementNB": ComplementNB,
            "BernoulliNB": BernoulliNB,
            "SGDClassifier": SGDClassifier,
            "RidgeClassifier": RidgeClassifier,
            "RandomForestClassifier": RandomForestClassifier,
            "GradientBoostingClassifier": GradientBoostingClassifier,
            "AdaBoostClassifier": AdaBoostClassifier,
            "LinearSVC": LinearSVC,
            "SVC": SVC,
            "KNeighborsClassifier": KNeighborsClassifier,
        }

    def set_max_iter(self, max_iter):
        """Set maximum number of iterations."""
        self.max_iter = max_iter

    def get_max_iter(self):
        """Return maximum number of iterations."""
        return self.max_iter

    def set_vc_clf_1(self, vc_clf_1):
        """Set 1st estimator for voting classifier."""
        self.vc_clf_1 = vc_clf_1

    def get_vc_clf_1(self):
        """Return 1st estimator for voting classifier."""
        return self.vc_clf_1

    def set_vc_clf_2(self, vc_clf_2):
        """Set 2nd estimator for voting classifier."""
        self.vc_clf_2 = vc_clf_2

    def get_vc_clf_2(self):
        """Return 2nd estimator for voting classifier."""
        return self.vc_clf_2

    def set_vc_clf_3(self, vc_clf_3):
        """Set 3rd estimator for voting classifier."""
        self.vc_clf_3 = vc_clf_3

    def get_vc_clf_3(self):
        """Return 3rd estimator for voting classifier."""
        return self.vc_clf_3

    def set_voting_hard(self):
        """Set voting option as 'hard' for classifier."""
        self.voting = "hard"

    def set_voting_soft(self):
        """Set voting option as 'soft' for classifier."""
        self.voting = "soft"

    def set_sc_clf_1(self, sc_clf_1):
        """Set 1st estimator for scoring classifier."""
        self.sc_clf_1 = sc_clf_1

    def get_sc_clf_1(self):
        """Return 1st estimator for scoring classifier."""
        return self.sc_clf_1

    def set_sc_clf_2(self, sc_clf_2):
        """Set 2nd estimator for scoring classifier."""
        self.sc_clf_2 = sc_clf_2

    def get_sc_clf_2(self):
        """Return 2nd estimator for scoring classifier."""
        return self.sc_clf_2

    def set_sc_clf_3(self, sc_clf_3):
        """Set 3rd estimator for scoring classifier."""
        self.sc_clf_3 = sc_clf_3

    def get_sc_clf_3(self):
        """Return 3rd estimator for scoring classifier."""
        return self.sc_clf_3

    def set_classifier(self, classifier):
        """Method that can set classifier."""
        self.classifier = classifier

    def get_classifier(self):
        """Method that returns classifier."""
        return self.classifier

    def set_clf_nb(self):
        """Method that can set classifier as MultinomialNB."""
        self.set_classifier(self.ESTIMATORS_AND_CLASSIFIERS["MultinomialNB"]())

    def set_clf_cnb(self):
        """Method that can set classifier as ComplementNB."""
        self.set_classifier(self.ESTIMATORS_AND_CLASSIFIERS["ComplementNB"]())

    def set_clf_bnb(self):
        """Method that can set classifier as BernoulliNB."""
        self.set_classifier(self.ESTIMATORS_AND_CLASSIFIERS["BernoulliNB"]())

    def set_clf_lr(self):
        """Method that can set classifier as LogisticRegression."""
        self.set_classifier(self.ESTIMATORS_AND_CLASSIFIERS["LogisticRegression"]())

    def set_clf_sgd(self):
        """Method that can set classifier as SGDClassifier."""
        self.set_classifier(self.ESTIMATORS_AND_CLASSIFIERS["SGDClassifier"]())

    def set_clf_rdg(self):
        """Method that can set classifier as RidgeClassifier."""
        self.set_classifier(self.ESTIMATORS_AND_CLASSIFIERS["RidgeClassifier"]())

    def set_clf_rfc(self):
        """Method that can set classifier as RandomForestClassifier."""
        self.set_classifier(self.ESTIMATORS_AND_CLASSIFIERS["RandomForestClassifier"]())

    def set_clf_gbc(self):
        """Method that can set classifier as GradientBoostingClassifier."""
        self.set_classifier(
            self.ESTIMATORS_AND_CLASSIFIERS["GradientBoostingClassifier"]()
        )

    def set_clf_abc(self):
        """Method that can set classifier as AdaBoostClassifier."""
        self.set_classifier(self.ESTIMATORS_AND_CLASSIFIERS["AdaBoostClassifier"])

    def set_clf_lsv(self):
        """Method that can set classifier as LinearSVC."""
        self.set_classifier(self.ESTIMATORS_AND_CLASSIFIERS["LinearSVC"]())

    def set_clf_svc(self):
        """Method that can set classifier as SVC."""
        self.set_classifier(self.ESTIMATORS_AND_CLASSIFIERS["SVC"]())

    def set_clf_knn(self):
        """Method that can set classifier as KNeighborsClassifier."""
        self.set_classifier(self.ESTIMATORS_AND_CLASSIFIERS["KNeighborsClassifier"])

    def set_clf_vtc(self):
        """Method that can set classifier as VotingClassifier."""
        self.set_classifier(
            VotingClassifier(
                estimators=[
                    ("1", self.get_vc_clf_1()),
                    ("2", self.get_vc_clf_2()),
                    ("3", self.get_vc_clf_3()),
                ],
                voting=self.voting,
            )
        )

    def set_clf_stc(self):
        """Method that can set classifier as StackingClassifier."""
        self.set_classifier(
            StackingClassifier(
                estimators=[
                    ("1", self.get_sc_clf_1()),
                    ("2", self.get_vc_clf_2()),
                    ("3", self.get_vc_clf_3()),
                ]
            )
        )
