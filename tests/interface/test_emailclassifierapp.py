import unittest
from unittest.mock import patch, MagicMock
from src.classifiers.classifier import Classifier
from src.data_module.dataloader import DataLoader
from src.interface.emailclassifierapp import EmailClassifierApp
from src.models.model import Model


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

        self.assertEqual(
            self.assertIsInstance(data_loader, DataLoader),
            self.assertIsInstance(app.data_loader, DataLoader),
        )

        self.assertEqual(
            self.assertIsNone(model1),
            self.assertIsNone(app.model1),
        )

        self.assertEqual(
            self.assertIsNone(model2),
            self.assertIsNone(app.model2),
        )

        self.assertEqual(
            self.assertIsNone(model3),
            self.assertIsNone(app.model3),
        )

        self.assertEqual(
            self.assertIsNone(multiclassifier_model),
            self.assertIsNone(app.multiclassifier_model),
        )
        del app

    def test_set_model1_clf(self):
        """Method tests the set_model1_clf method of the class."""
        app = EmailClassifierApp()
        output = app.set_model1_clf("ExampleClf")
        self.assertEqual(app.model1.clf, "ExampleClf")
        self.assertIsInstance(output, Model)
        del app

    def test_set_model2_clf(self):
        """Method tests the set_model2_clf method of the class."""
        app = EmailClassifierApp()
        output = app.set_model2_clf("ExampleClf")
        self.assertEqual(app.model2.clf, "ExampleClf")
        self.assertIsInstance(output, Model)
        del app

    def test_set_model3_clf(self):
        """Method tests the set_model3_clf method of the class."""
        app = EmailClassifierApp()
        output = app.set_model3_clf("ExampleClf")
        self.assertEqual(app.model3.clf, "ExampleClf")
        self.assertIsInstance(output, Model)
        del app

    def test_set_multiclassifier_model_clf(self):
        """Method tests the set_multiclassifier_model_clf method of the class."""
        app = EmailClassifierApp()
        output = app.set_multiclassifier_model_clf("ExampleClf")
        self.assertEqual(app.multiclassifier_model.clf, "ExampleClf")
        self.assertIsInstance(output, Model)
        del app

    def test_load_data_csv_exception(self):
        """Method tests the load_data_csv method of the class, in the environment that returns an exception."""
        app = EmailClassifierApp()
        result = app.load_data_csv("./nonexistentpath/test.csv")
        self.assertIsNone(result)
        del app

    def test_load_data_csv_success(self):
        """Test that checks if load_data_csv returns a DataFrame when given a valid path."""

        app = EmailClassifierApp()
        fake_path = "fake/path/to/data.csv"
        fake_df = MagicMock()

        with patch.object(app.data_loader, "set_path") as mock_set_path, \
                patch.object(app.data_loader, "load_data_csv", return_value=fake_df) as mock_load:
            result = app.load_data_csv(fake_path)

            # Ensures set_path() was called correctly
            mock_set_path.assert_called_once_with(fake_path)

            # Ensures data_loader.load_data_csv() was called
            mock_load.assert_called_once()

            # Ensures returned DF is exactly what mocked loader returned
            self.assertIs(result, fake_df)
        del app

    def test_classifier_option_check_true(self):
        """Method tests the classifier_option_check method of the class when it returns true."""
        app = EmailClassifierApp()
        option = "Example1"
        const = {"Example1": "Example1", "Example2": "Example2"}
        result = app.classifier_option_check(option, const)
        self.assertTrue(result)
        del app

    def test_classifier_option_check_false(self):
        """Method tests the classifier_option_check method of the class when it returns false."""
        app = EmailClassifierApp()
        option = "Example3"
        const = {"Example1": "Example1", "Example2": "Example2"}
        result = app.classifier_option_check(option, const)
        self.assertFalse(result)
        del app

    def test_train_3_stage_pipelines_invalid_classifier(self):
        """Method tests the train_3_stage_pipelines method of the class when the provided classifier is invalid."""

        app = EmailClassifierApp()

        with patch.object(app, "load_data_csv", return_value=MagicMock()), \
                patch.object(app, "classifier_option_check", return_value=False):
            result = app.train_3_stage_pipelines(
                classifier_option_1="BadClf",
            )

            # Models set to None
            self.assertIsNone(app.model1)
            self.assertIsNone(app.model2)
            self.assertIsNone(app.model3)
            self.assertIsNone(result)

    def test_train_3_stage_voting_classifier_not_supported(self):
        """Method tests the train_3_stage_pipelines method of the class when the provided classifier is an unsupported VotingClassifier."""

        app = EmailClassifierApp()

        fake_df = MagicMock()
        fake_model = MagicMock()

        app.set_model1_clf = MagicMock(return_value=fake_model)
        app.set_model2_clf = MagicMock(return_value=fake_model)
        app.set_model3_clf = MagicMock(return_value=fake_model)

        app.classifier = MagicMock()
        app.classifier.get_classifier = MagicMock(return_value="FAKE_CLF")

        with patch.object(app, "load_data_csv", return_value=fake_df), \
                patch.object(app, "classifier_option_check", return_value=True), \
                patch.object(app, "CLASSIFIERS", {"MultinomialNB": MagicMock()}) as mock_clf_dict:
            app.train_3_stage_pipelines(
                classifier_option_1="VotingClassifier",
                classifier_option_2="VotingClassifier",
                classifier_option_3="VotingClassifier",
            )

            # MultinomialNB should be called as fallback 3×
            self.assertEqual(mock_clf_dict["MultinomialNB"].call_count, 3)

    def test_train_3_stage_stacking_classifier_not_supported(self):
        """Method tests the train_3_stage_pipelines method of the class when the provided classifier is an unsupported StackingClassifier."""

        app = EmailClassifierApp()

        fake_df = MagicMock()
        fake_model = MagicMock()

        app.set_model1_clf = MagicMock(return_value=fake_model)
        app.set_model2_clf = MagicMock(return_value=fake_model)
        app.set_model3_clf = MagicMock(return_value=fake_model)

        app.classifier = MagicMock()
        app.classifier.get_classifier = MagicMock(return_value="FAKE_CLF")

        with patch.object(app, "load_data_csv", return_value=fake_df), \
                patch.object(app, "classifier_option_check", return_value=True), \
                patch.object(app, "CLASSIFIERS", {"MultinomialNB": MagicMock()}) as mock_clf_dict:
            app.train_3_stage_pipelines(
                classifier_option_1="StackingClassifier",
                classifier_option_2="StackingClassifier",
                classifier_option_3="StackingClassifier",
            )

            self.assertEqual(mock_clf_dict["MultinomialNB"].call_count, 3)

    def test_train_3_stage_pipelines_success(self):
        """Method tests the train_3_stage_pipelines method of the class."""

        app = EmailClassifierApp()

        # Fake dataframe with required columns
        fake_df = {
            "email_text": ["text1", "text2"],
            "related_to_jobhunt": [1, 0],
            "is_confirmation": [0, 1],
            "is_invitation": [1, 0],
        }
        fake_df = MagicMock()
        fake_df.__getitem__.side_effect = lambda key: [1, 0]

        # Fake pipeline model
        fake_model = MagicMock()
        fake_model.build_pipeline = MagicMock()
        fake_model.set_X = MagicMock()
        fake_model.set_y = MagicMock()
        fake_model.train = MagicMock()

        # Mock setter functions set_model1_clf etc.
        app.set_model1_clf = MagicMock(return_value=fake_model)
        app.set_model2_clf = MagicMock(return_value=fake_model)
        app.set_model3_clf = MagicMock(return_value=fake_model)

        # Fake classifier setter
        app.classifier = MagicMock()
        app.classifier.get_classifier = MagicMock(return_value="FAKE_CLF")

        with patch.object(app, "load_data_csv", return_value=fake_df), \
                patch.object(app, "classifier_option_check", return_value=True), \
                patch.object(app, "CLASSIFIERS", {"MultinomialNB": MagicMock()}):
            app.train_3_stage_pipelines(
                path1="p1",
                path2="p2",
                path3="p3",
            )

            # 3 calls to load_data_csv
            self.assertEqual(app.load_data_csv.call_count, 3)

            # 3 classifier setters called
            app.set_model1_clf.assert_called_once()
            app.set_model2_clf.assert_called_once()
            app.set_model3_clf.assert_called_once()

            # Check training was triggered 3 times
            self.assertEqual(fake_model.train.call_count, 3)

    def test_view_3_stage_pipelines_accuracy_model_not_found(self):
        """Method tests if view_3_stage_pipelines_accuracy returns None when any model is missing."""

        app = EmailClassifierApp()

        # Case: model1 is None → triggers ModelNotFound
        app.model1 = None
        app.model2 = MagicMock()
        app.model3 = MagicMock()

        result = app.view_3_stage_pipelines_accuracy()

        self.assertIsNone(result)

    def test_view_3_stage_pipelines_accuracy_success(self):
        """Method tests the success route for view_3_stage_pipelines_accuracy where accuracy of all 3 models is displayed."""

        app = EmailClassifierApp()

        # Fake accuracy values for count_accuracy()
        fake_accuracy = (0.85, 0.90)

        # Create three fake model objects
        model1 = MagicMock()
        model2 = MagicMock()
        model3 = MagicMock()

        model1.count_accuracy.return_value = fake_accuracy
        model2.count_accuracy.return_value = fake_accuracy
        model3.count_accuracy.return_value = fake_accuracy

        app.model1 = model1
        app.model2 = model2
        app.model3 = model3

        # Patch print to avoid console output during tests
        with patch("builtins.print"):
            result = app.view_3_stage_pipelines_accuracy()

        # Ensure accuracy was computed for each model
        model1.count_accuracy.assert_called_once()
        model2.count_accuracy.assert_called_once()
        model3.count_accuracy.assert_called_once()

        # Method does not return accuracy, only prints → should be None
        self.assertIsNone(result)

    def test_classify_3_stage_model1_not_found(self):
        """Method tests the classify_emails_3_stage_pipelines when model1 is missing."""
        app = EmailClassifierApp()

        app.model1 = None  # missing → triggers exception
        app.model2 = MagicMock()
        app.model3 = MagicMock()

        result = app.classify_emails_3_stage_pipelines("test email")
        self.assertIsNone(result)

    def test_classify_3_stage_model2_not_found(self):
        """Method tests the classify_emails_3_stage_pipelines when model2 is missing."""
        app = EmailClassifierApp()

        app.model1 = MagicMock()
        app.model2 = None
        app.model3 = MagicMock()

        result = app.classify_emails_3_stage_pipelines("test email")
        self.assertIsNone(result)

    def test_classify_3_stage_model3_not_found(self):
        """Method tests the classify_emails_3_stage_pipelines when model3 is missing."""
        app = EmailClassifierApp()

        app.model1 = MagicMock()
        app.model2 = MagicMock()
        app.model3 = None

        result = app.classify_emails_3_stage_pipelines("test email")
        self.assertIsNone(result)

    def test_classify_3_stage_models_not_found(self):
        """Method tests the classify_emails_3_stage_pipelines when models are missing."""
        app = EmailClassifierApp()

        app.model1 = None
        app.model2 = None
        app.model3 = None

        result = app.classify_emails_3_stage_pipelines("test email")
        self.assertIsNone(result)

    def test_classify_3_stage_input_error_not_list_or_str(self):
        """Method tests the classify_emails_3_stage_pipelines when input is not str or list."""
        app = EmailClassifierApp()

        # All models mocked
        app.model1 = MagicMock()
        app.model2 = MagicMock()
        app.model3 = MagicMock()

        with patch("builtins.print"):
            result = app.classify_emails_3_stage_pipelines(12345)  # invalid type

        self.assertIsNone(result)

    def test_classify_3_stage_input_error_non_string_inside_list(self):
        """Method tests the classify_emails_3_stage_pipelines when list contains non-string elements."""
        app = EmailClassifierApp()

        app.model1 = MagicMock()
        app.model2 = MagicMock()
        app.model3 = MagicMock()

        with patch("builtins.print"):
            result = app.classify_emails_3_stage_pipelines(["ok", 999])  # 999 invalid

        self.assertIsNone(result)

if __name__ == "__main__":
    unittest.main()
