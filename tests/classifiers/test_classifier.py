import unittest
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier, VotingClassifier, StackingClassifier,
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
        del clf

    def test_get_classifier(self):
        """Method tests the get_classifier method of the class."""
        clf = Classifier()
        clf.classifier = "classifier"
        self.assertEqual("classifier", clf.get_classifier())
        del clf

    def test_set_clf_nb(self):
        """Method tests the set_clf_nb method of the class."""
        clf = Classifier()
        classifier = MultinomialNB()
        clf.set_clf_nb()
        self.assertEqual(
            self.assertIsInstance(classifier, MultinomialNB),
            self.assertIsInstance(clf.classifier, MultinomialNB),
        )
        del clf

    def test_set_clf_cnb(self):
        """Method tests the set_clf_cnb method of the class."""
        clf = Classifier()
        classifier = ComplementNB()
        clf.set_clf_cnb()
        self.assertEqual(
            self.assertIsInstance(classifier, ComplementNB),
            self.assertIsInstance(clf.classifier, ComplementNB),
        )
        del clf

    def test_set_clf_bnb(self):
        """Method tests the set_clf_bnb method of the class."""
        clf = Classifier()
        classifier = BernoulliNB()
        clf.set_clf_bnb()
        self.assertEqual(
            self.assertIsInstance(classifier, BernoulliNB),
            self.assertIsInstance(clf.classifier, BernoulliNB),
        )
        del clf

    def test_set_clf_lr(self):
        """Method tests the set_clf_lr method of the class."""
        clf = Classifier()
        classifier = LogisticRegression()
        clf.set_clf_lr()
        self.assertEqual(
            self.assertIsInstance(classifier, LogisticRegression),
            self.assertIsInstance(clf.classifier, LogisticRegression),
        )
        del clf

    def test_set_clf_sgd(self):
        """Method tests the set_clf_sgd method of the class."""
        clf = Classifier()
        classifier = SGDClassifier()
        clf.set_clf_sgd()
        self.assertEqual(
            self.assertIsInstance(classifier, SGDClassifier),
            self.assertIsInstance(clf.classifier, SGDClassifier),
        )
        del clf

    def test_set_clf_rdg(self):
        """Method tests the set_clf_rdg method of the class."""
        clf = Classifier()
        classifier = RidgeClassifier()
        clf.set_clf_rdg()
        self.assertEqual(
            self.assertIsInstance(classifier, RidgeClassifier),
            self.assertIsInstance(clf.classifier, RidgeClassifier),
        )
        del clf

    def test_set_clf_rfc(self):
        """Method tests the set_clf_rfc method of the class."""
        clf = Classifier()
        classifier = RandomForestClassifier()
        clf.set_clf_rfc()
        self.assertEqual(
            self.assertIsInstance(classifier, RandomForestClassifier),
            self.assertIsInstance(clf.classifier, RandomForestClassifier),
        )
        del clf

    def test_set_clf_gbc(self):
        """Method tests the set_clf_gbc method of the class."""
        clf = Classifier()
        classifier = GradientBoostingClassifier()
        clf.set_clf_gbc()
        self.assertEqual(
            self.assertIsInstance(classifier, GradientBoostingClassifier),
            self.assertIsInstance(clf.classifier, GradientBoostingClassifier),
        )
        del clf

    def test_set_clf_abc(self):
        """Method tests the set_clf_abc method of the class."""
        clf = Classifier()
        classifier = AdaBoostClassifier()
        clf.set_clf_abc()
        self.assertEqual(
            self.assertIsInstance(classifier, AdaBoostClassifier),
            self.assertIsInstance(clf.classifier, AdaBoostClassifier),
        )
        del clf

    def test_set_clf_lsv(self):
        """Method tests the set_clf_lsv method of the class."""
        clf = Classifier()
        classifier = LinearSVC()
        clf.set_clf_lsv()
        self.assertEqual(
            self.assertIsInstance(classifier, LinearSVC),
            self.assertIsInstance(clf.classifier, LinearSVC),
        )
        del clf

    def test_set_clf_svc(self):
        """Method tests the set_clf_svc method of the class."""
        clf = Classifier()
        classifier = SVC()
        clf.set_clf_svc()
        self.assertEqual(
            self.assertIsInstance(classifier, SVC),
            self.assertIsInstance(clf.classifier, SVC),
        )
        del clf

    def test_set_clf_knn(self):
        """Method tests the set_clf_knn method of the class."""
        clf = Classifier()
        classifier = KNeighborsClassifier()
        clf.set_clf_knn()
        self.assertEqual(
            self.assertIsInstance(classifier, KNeighborsClassifier),
            self.assertIsInstance(clf.classifier, KNeighborsClassifier),
        )
        del clf

    def test_set_clf_dtc(self):
        """Method tests the set_clf_dtc method of the class."""
        clf = Classifier()
        classifier = DecisionTreeClassifier()
        clf.set_clf_dtc()
        self.assertEqual(
            self.assertIsInstance(classifier, DecisionTreeClassifier),
            self.assertIsInstance(clf.classifier, DecisionTreeClassifier),
        )
        del clf

    def test_set_clf_etc(self):
        """Method tests the set_clf_etc method of the class."""
        clf = Classifier()
        classifier = ExtraTreeClassifier()
        clf.set_clf_etc()
        self.assertEqual(
            self.assertIsInstance(classifier, ExtraTreeClassifier),
            self.assertIsInstance(clf.classifier, ExtraTreeClassifier),
        )
        del clf

    def test_set_clf_vtc(self):
        """Method tests the set_clf_vtc method of the class."""
        clf = Classifier()
        classifier = VotingClassifier(
            estimators=[
                ("1", LogisticRegression(max_iter=200)),
                ("2", MultinomialNB()),
                ("3", LinearSVC()),
            ],
            voting="hard",
        )
        clf.set_clf_vtc()
        self.assertEqual(
            self.assertIsInstance(classifier, VotingClassifier),
            self.assertIsInstance(clf.classifier, VotingClassifier),
        )
        del clf

    def test_set_clf_stc(self):
        """Method tests the set_clf_stc method of the class."""
        clf = Classifier()
        classifier = StackingClassifier(
            estimators=[
                ("1", MultinomialNB()),
                ("2", LinearSVC()),
                ("3", RandomForestClassifier()),
            ]
        )
        clf.set_clf_stc()
        self.assertEqual(
            self.assertIsInstance(classifier, StackingClassifier),
            self.assertIsInstance(clf.classifier, StackingClassifier),
        )
        del clf


if __name__ == "__main__":
    unittest.main()
