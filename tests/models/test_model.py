import unittest
from src.models.model import Model
from src.classifiers.classifier import Classifier

class TestModel(unittest.TestCase):

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

if __name__ == "__main__":
    unittest.main()