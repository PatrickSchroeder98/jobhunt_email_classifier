import unittest
from sklearn.naive_bayes import MultinomialNB
from src.models.model import Model
from src.classifiers.classifier import Classifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer


class TestModel(unittest.TestCase):
    """Class with tests associated with Model class."""

    def test_init(self):
        """Method tests the initialization of a model"""
        clf = Classifier()
        model = Model(clf)
        self.assertEqual(clf, model.clf)
        self.assertEqual(model.y_pred, None)
        self.assertEqual(model.pipeline, None)
        self.assertEqual(model.X, None)
        self.assertEqual(model.y, None)
        self.assertEqual(model.test_size, 0.3)
        self.assertEqual(model.random_state, 42)
        self.assertEqual(model.X_train, None)
        self.assertEqual(model.y_train, None)
        self.assertEqual(model.X_test, None)
        self.assertEqual(model.y_test, None)

        del clf, model

    def test_set_clf(self):
        """Method tests the set_clf method of the model."""
        clf = Classifier()
        clf2 = Classifier()
        model = Model(clf)

        model.set_clf(clf2)
        self.assertEqual(model.clf, clf2)

        del clf, clf2, model

    def test_get_clf(self):
        """Method tests the get_clf method of the model."""
        clf = Classifier()
        model = Model(clf)
        self.assertEqual(model.get_clf(), clf)

        del clf, model

    def test_set_X_train(self):
        """Method tests the set_X_train method of the model."""
        clf = Classifier()
        model = Model(clf)

        model.set_X_train([0, 1, 2])
        self.assertEqual(model.X_train, [0, 1, 2])

        del clf, model

    def test_get_X_train(self):
        """Method tests the get_X_train method of the model."""
        clf = Classifier()
        model = Model(clf)

        model.X_train = [0, 1, 2]
        self.assertEqual(model.get_X_train(), [0, 1, 2])

        del clf, model

    def test_set_y_train(self):
        """Method tests the set_y_train method of the model."""
        clf = Classifier()
        model = Model(clf)

        model.set_y_train([0, 1, 2])
        self.assertEqual(model.y_train, [0, 1, 2])

        del clf, model

    def test_get_y_train(self):
        """Method tests the get_y_train method of the model."""
        clf = Classifier()
        model = Model(clf)

        model.y_train = [0, 1, 2]
        self.assertEqual(model.get_y_train(), [0, 1, 2])

        del clf, model

    def test_set_X_test(self):
        """Method tests the set_X_test method of the model."""
        clf = Classifier()
        model = Model(clf)

        model.set_X_test([0, 1, 2])
        self.assertEqual(model.X_test, [0, 1, 2])

        del clf, model

    def test_get_X_test(self):
        """Method tests the get_X_test method of the model."""
        clf = Classifier()
        model = Model(clf)

        model.X_test = [0, 1, 2]
        self.assertEqual(model.get_X_test(), [0, 1, 2])

        del clf, model

    def test_set_y_test(self):
        """Method tests the set_y_test method of the model."""
        clf = Classifier()
        model = Model(clf)

        model.set_y_test([0, 1, 2])
        self.assertEqual(model.y_test, [0, 1, 2])

        del clf, model

    def test_get_y_test(self):
        """Method tests the get_y_test method of the model."""
        clf = Classifier()
        model = Model(clf)

        model.y_test = [0, 1, 2]
        self.assertEqual(model.get_y_test(), [0, 1, 2])

        del clf, model

    def test_set_X(self):
        """Method tests the set_X method of the model."""
        clf = Classifier()
        model = Model(clf)

        model.set_X([0, 1, 2])
        self.assertEqual(model.X, [0, 1, 2])

        del clf, model

    def test_get_X(self):
        """Method tests the get_X method of the model."""
        clf = Classifier()
        model = Model(clf)

        model.X = [0, 1, 2]
        self.assertEqual(model.get_X(), [0, 1, 2])

        del clf, model

    def test_set_y(self):
        """Method tests the set_y method of the model."""
        clf = Classifier()
        model = Model(clf)

        model.set_y([0, 1, 2])
        self.assertEqual(model.y, [0, 1, 2])

        del clf, model

    def test_get_y(self):
        """Method tests the get_y method of the model."""
        clf = Classifier()
        model = Model(clf)

        model.y = [0, 1, 2]
        self.assertEqual(model.get_y(), [0, 1, 2])

        del clf, model

    def test_set_test_size(self):
        """Method tests the set_test_size method of the model."""
        clf = Classifier()
        model = Model(clf)

        model.set_test_size(0.3)
        self.assertEqual(model.test_size, 0.3)

        del clf, model

    def test_get_test_size(self):
        """Method tests the get_test_size method of the model."""
        clf = Classifier()
        model = Model(clf)

        model.test_size = 0.3
        self.assertEqual(model.get_test_size(), 0.3)

        del clf, model

    def test_get_pipeline(self):
        """Method tests the get_pipeline method of the model."""
        clf = Classifier()
        clf.set_clf_nb()
        model = Model(clf)

        p = Pipeline([("tfidf", TfidfVectorizer()), ("classifier", model.clf)])
        model.pipeline = p

        self.assertEqual(model.get_pipeline(), p)

    def test_set_random_state(self):
        """Method tests the set_random_state method of the model."""
        clf = Classifier()
        model = Model(clf)

        model.set_random_state(42)
        self.assertEqual(model.random_state, 42)

        del clf, model

    def test_get_random_state(self):
        """Method tests the get_random_state method of the model."""
        clf = Classifier()
        model = Model(clf)

        model.random_state = 42
        self.assertEqual(model.get_random_state(), 42)

        del clf, model

    def test_set_y_pred(self):
        """Method tests the set_y_pred method of the model."""
        clf = Classifier()
        model = Model(clf)

        model.set_y_pred([0, 1, 2])
        self.assertEqual(model.y_pred, [0, 1, 2])

        del clf, model

    def test_get_y_pred(self):
        """Method tests the get_y_pred method of the model."""
        clf = Classifier()
        model = Model(clf)

        model.y_pred = [0, 1, 2]
        self.assertEqual(model.get_y_pred(), [0, 1, 2])

        del clf, model

    def test_build_pipeline(self):
        """Method tests the build_pipeline method of the model."""
        clf = Classifier()
        clf.set_clf_nb()
        model = Model(clf.classifier)

        self.assertIsNone(model.pipeline)
        model.build_pipeline()
        self.assertIsNotNone(model.pipeline)
        self.assertEqual(model.pipeline.steps[0][0], "tfidf")
        self.assertIsInstance(model.pipeline.steps[0][1], TfidfVectorizer)
        self.assertEqual(model.pipeline.steps[1][0], "classifier")
        self.assertIsInstance(model.pipeline.steps[1][1], MultinomialNB)

        del clf, model

    def test_train(self):
        """Method tests the train method of the model."""
        clf = Classifier()
        clf.set_clf_nb()
        model = Model(clf.classifier)

        model.X = [
            "This is jobhunt related",
            "Y",
            "This is jobhunt related",
            "Y",
            "This is jobhunt related",
            "Y",
        ]
        model.y = ["True", "False", "True", "False", "True", "False"]
        model.build_pipeline()
        self.assertIsNone(model.X_train)
        self.assertIsNone(model.X_test)
        self.assertIsNone(model.y_train)
        self.assertIsNone(model.y_test)
        model.train()
        self.assertIsNotNone(model.X_train)
        self.assertIsNotNone(model.X_test)
        self.assertIsNotNone(model.y_train)
        self.assertIsNotNone(model.y_test)

        del clf, model

    def test_count_accuracy(self):
        """Method tests the count_accuracy method of the model."""
        clf = Classifier()
        clf.set_clf_nb()
        model = Model(clf.classifier)
        model.X = [
            "This is jobhunt related",
            "Y",
            "This is jobhunt related",
            "Y",
            "This is jobhunt related",
            "Y",
        ]
        model.y = ["True", "False", "True", "False", "True", "False"]
        model.build_pipeline()
        model.train()
        result = model.count_accuracy()

        # --- structural tests ---
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

        # --- check accuracy value ---
        self.assertIsInstance(result[0], float)
        self.assertGreaterEqual(result[0], 0.0)
        self.assertLessEqual(result[0], 1.0)

        # --- check classification report ---
        self.assertIsInstance(result[1], str)
        self.assertIn("precision", result[1])
        self.assertIn("recall", result[1])
        self.assertIn("f1-score", result[1])

        del clf, model


if __name__ == "__main__":
    unittest.main()
