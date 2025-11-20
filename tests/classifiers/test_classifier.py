import unittest
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
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
        self.assertEqual(self.assertIsInstance(vc_clf_1, LogisticRegression), self.assertIsInstance(clf.vc_clf_1, LogisticRegression))
        self.assertEqual(vc_clf_2, clf.vc_clf_2)
        self.assertEqual(vc_clf_3, clf.vc_clf_3)
        self.assertEqual(voting, clf.voting)
        self.assertEqual(sc_clf_1, clf.sc_clf_1)
        self.assertEqual(sc_clf_2, clf.sc_clf_2)
        self.assertEqual(sc_clf_3, clf.sc_clf_3)
        self.assertEqual(ESTIMATORS_AND_CLASSIFIERS, clf.ESTIMATORS_AND_CLASSIFIERS)

        del clf

if __name__ == "__main__":
    unittest.main()
