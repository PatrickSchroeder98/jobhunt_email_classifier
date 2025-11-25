import unittest
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn.naive_bayes import MultinomialNB, ComplementNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from src.classifiers.classifier import Classifier


class TestClassifier(unittest.TestCase):
    """Class with tests associated with Classifier class."""

    def test_init(self):
        """Method tests the initialization of a class"""
        clf = Classifier()

        classifier = None
        max_iter = 200
        vc_clf_1 = LogisticRegression(max_iter=max_iter)
        vc_clf_2 = MultinomialNB()
        vc_clf_3 = LinearSVC()
        voting = "hard"

        sc_clf_1 = MultinomialNB()
        sc_clf_2 = LinearSVC()
        sc_clf_3 = RandomForestClassifier()

        ESTIMATORS_AND_CLASSIFIERS = {
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
            "DecisionTreeClassifier": DecisionTreeClassifier,
            "ExtraTreeClassifier": ExtraTreeClassifier,
        }

        self.assertIsInstance(clf, Classifier)
        self.assertEqual(classifier, clf.classifier)
        self.assertEqual(max_iter, clf.max_iter)
        self.assertEqual(
            self.assertIsInstance(vc_clf_1, LogisticRegression),
            self.assertIsInstance(clf.vc_clf_1, LogisticRegression),
        )
        self.assertEqual(
            self.assertIsInstance(vc_clf_2, MultinomialNB),
            self.assertIsInstance(clf.vc_clf_2, MultinomialNB),
        )
        self.assertEqual(
            self.assertIsInstance(vc_clf_3, LinearSVC),
            self.assertIsInstance(clf.vc_clf_3, LinearSVC),
        )
        self.assertEqual(voting, clf.voting)
        self.assertEqual(
            self.assertIsInstance(sc_clf_1, MultinomialNB),
            self.assertIsInstance(clf.sc_clf_1, MultinomialNB),
        )
        self.assertEqual(
            self.assertIsInstance(sc_clf_2, LinearSVC),
            self.assertIsInstance(clf.sc_clf_2, LinearSVC),
        )
        self.assertEqual(
            self.assertIsInstance(sc_clf_3, RandomForestClassifier),
            self.assertIsInstance(clf.sc_clf_3, RandomForestClassifier),
        )
        self.assertEqual(ESTIMATORS_AND_CLASSIFIERS, clf.ESTIMATORS_AND_CLASSIFIERS)

        del clf

    def test_set_max_iter(self):
        """Method tests the set_max_iter method of the class."""
        clf = Classifier()
        clf.set_max_iter(100)
        self.assertEqual(100, clf.max_iter)
        del clf

    def test_get_max_iter(self):
        """Method tests the get_max_iter method of the class."""
        clf = Classifier()
        clf.max_iter = 150
        self.assertEqual(150, clf.get_max_iter())
        del clf

    def test_set_vc_clf_1(self):
        """Method tests the set_vc_clf_1 method of the class."""
        clf = Classifier()
        clf.set_vc_clf_1("estimator1")
        self.assertEqual("estimator1", clf.vc_clf_1)
        del clf

    def test_get_vc_clf_1(self):
        """Method tests the get_vc_clf_1 method of the class."""
        clf = Classifier()
        clf.vc_clf_1 = "estimator1"
        self.assertEqual("estimator1", clf.get_vc_clf_1())
        del clf

    def test_set_vc_clf_2(self):
        """Method tests the set_vc_clf_2 method of the class."""
        clf = Classifier()
        clf.set_vc_clf_2("estimator2")
        self.assertEqual("estimator2", clf.vc_clf_2)
        del clf

    def test_get_vc_clf_2(self):
        """Method tests the get_vc_clf_2 method of the class."""
        clf = Classifier()
        clf.vc_clf_2 = "estimator2"
        self.assertEqual("estimator2", clf.get_vc_clf_2())
        del clf

    def test_set_vc_clf_3(self):
        """Method tests the set_vc_clf_3 method of the class."""
        clf = Classifier()
        clf.set_vc_clf_3("estimator3")
        self.assertEqual("estimator3", clf.vc_clf_3)
        del clf

    def test_get_vc_clf_3(self):
        """Method tests the get_vc_clf_2 method of the class."""
        clf = Classifier()
        clf.vc_clf_3 = "estimator3"
        self.assertEqual("estimator3", clf.get_vc_clf_3())
        del clf

    def test_set_voting_hard(self):
        """Method tests the set_voting_hard method of the class."""
        clf = Classifier()
        clf.set_voting_hard()
        self.assertEqual("hard", clf.voting)
        del clf

    def test_set_voting_soft(self):
        """Method tests the set_voting_soft method of the class."""
        clf = Classifier()
        clf.set_voting_soft()
        self.assertEqual("soft", clf.voting)
        del clf

    def test_set_sc_clf_1(self):
        """Method tests the set_sc_clf_1 method of the class."""
        clf = Classifier()
        clf.set_sc_clf_1("estimator1")
        self.assertEqual("estimator1", clf.sc_clf_1)
        del clf

    def test_get_sc_clf_1(self):
        """Method tests the get_sc_clf_1 method of the class."""
        clf = Classifier()
        clf.sc_clf_1 = "estimator1"
        self.assertEqual("estimator1", clf.get_sc_clf_1())
        del clf

    def test_set_sc_clf_2(self):
        """Method tests the set_sc_clf_2 method of the class."""
        clf = Classifier()
        clf.set_sc_clf_2("estimator2")
        self.assertEqual("estimator2", clf.sc_clf_2)
        del clf

    def test_get_sc_clf_2(self):
        """Method tests the get_sc_clf_2 method of the class."""
        clf = Classifier()
        clf.sc_clf_2 = "estimator2"
        self.assertEqual("estimator2", clf.get_sc_clf_2())
        del clf

    def test_set_sc_clf_3(self):
        """Method tests the set_sc_clf_3 method of the class."""
        clf = Classifier()
        clf.set_sc_clf_3("estimator3")
        self.assertEqual("estimator3", clf.sc_clf_3)
        del clf

    def test_get_sc_clf_3(self):
        """Method tests the get_sc_clf_3 method of the class."""
        clf = Classifier()
        clf.sc_clf_3 = "estimator3"
        self.assertEqual("estimator3", clf.get_sc_clf_3())
        del clf

    def test_set_classifier(self):
        """Method tests the set_classifier method of the class."""
        clf = Classifier()
        clf.set_classifier("classifier")
        self.assertEqual("classifier", clf.classifier)

    def test_get_classifier(self):
        """Method tests the get_classifier method of the class."""
        clf = Classifier()
        clf.classifier = "classifier"
        self.assertEqual("classifier", clf.get_classifier())

if __name__ == "__main__":
    unittest.main()
