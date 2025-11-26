import unittest
from src.classifiers.classifier import Classifier
from src.data_module.dataloader import DataLoader
from src.interface.emailclassifierapp import EmailClassifierApp


class TestEmailClassifierApp(unittest.TestCase):
    """Class with tests associated with EmailClassifierApp class."""

    def test_init(self):
        """Method tests the initialization of a class"""
        app = EmailClassifierApp()
        data_loader = DataLoader()
        classifier = Classifier()
        model1 = None
        model2 = None
        model3 = None
        multiclassifier_model = None
        CLASSIFIERS = {
            "MultinomialNB": classifier.set_clf_nb,
            "LogisticRegression": classifier.set_clf_lr,
            "ComplementNB": classifier.set_clf_cnb,
            "BernoulliNB": classifier.set_clf_bnb,
            "SGDClassifier": classifier.set_clf_sgd,
            "RidgeClassifier": classifier.set_clf_rdg,
            "RandomForestClassifier": classifier.set_clf_rfc,
            "GradientBoostingClassifier": classifier.set_clf_gbc,
            "AdaBoostClassifier": classifier.set_clf_abc,
            "LinearSVC": classifier.set_clf_lsv,
            "SVC": classifier.set_clf_svc,
            "KNeighborsClassifier": classifier.set_clf_knn,
            "VotingClassifier": classifier.set_clf_vtc,
            "StackingClassifier": classifier.set_clf_stc,
            "DecisionTreeClassifier": classifier.set_clf_dtc,
            "ExtraTreeClassifier": classifier.set_clf_etc,
        }

        self.assertEqual(set(CLASSIFIERS.keys()), set(app.CLASSIFIERS.keys()))

        for key in CLASSIFIERS:
            expected_val = CLASSIFIERS[key]
            actual_val = app.CLASSIFIERS[key]
            self.assertIsInstance(actual_val, type(expected_val))

if __name__ == "__main__":
    unittest.main()
