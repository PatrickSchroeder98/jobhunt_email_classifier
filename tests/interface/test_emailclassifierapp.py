import unittest
import pandas as pd
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

    def test_classify_3_stage_model1_and_model2_not_found(self):
        """Method tests the classify_emails_3_stage_pipelines when model1 and model2 are missing."""
        app = EmailClassifierApp()

        app.model1 = None
        app.model2 = None
        app.model3 = MagicMock()

        result = app.classify_emails_3_stage_pipelines("test email")
        self.assertIsNone(result)

    def test_classify_3_stage_model1_and_model3_not_found(self):
        """Method tests the classify_emails_3_stage_pipelines when model1 and model3 are missing."""
        app = EmailClassifierApp()

        app.model1 = None
        app.model2 = MagicMock()
        app.model3 = None

        result = app.classify_emails_3_stage_pipelines("test email")
        self.assertIsNone(result)

    def test_classify_3_stage_model2_and_model3_not_found(self):
        """Method tests the classify_emails_3_stage_pipelines when model2 and model3 are missing."""
        app = EmailClassifierApp()

        app.model1 = MagicMock()
        app.model2 = None
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

    def test_classify_3_stage_success_all_paths(self):
        """Method tests the classify_emails_3_stage_pipelines with correct classification logic through all 3 stages using mocks."""

        app = EmailClassifierApp()

        # -------------------------
        # Mock models and pipelines
        # -------------------------
        model1 = MagicMock()
        model2 = MagicMock()
        model3 = MagicMock()

        app.model1 = model1
        app.model2 = model2
        app.model3 = model3

        # Prepare fake pipelines
        model1.pipeline = MagicMock()
        model2.pipeline = MagicMock()
        model3.pipeline = MagicMock()

        # --------------------
        # Input emails to test
        # --------------------
        emails = [
            "not job related",  # stage1 → False
            "confirmation mail",  # stage1 True, stage2 True
            "invitation mail",  # stage1 True, stage2 False, stage3 True
            "rejection mail",  # stage1 True, stage2 False, stage3 False
        ]

        # ---------------------------------------------
        # Define model predictions for each input email
        # ---------------------------------------------

        # Stage 1 predictions:
        # email0 → False
        # email1 → True
        # email2 → True
        # email3 → True
        model1.pipeline.predict.side_effect = [
            [False],  # email0
            [True],  # email1
            [True],  # email2
            [True],  # email3
        ]

        # Stage 2 predictions:
        # email1 → True
        # email2 → False
        # email3 → False
        model2.pipeline.predict.side_effect = [
            [True],  # email1
            [False],  # email2
            [False],  # email3
        ]

        # Stage 3 predictions:
        # email2 → True
        # email3 → False
        model3.pipeline.predict.side_effect = [
            [True],  # email2
            [False],  # email3
        ]

        # Run method
        with patch("builtins.print"):  # suppress prints
            results = app.classify_emails_3_stage_pipelines(emails)

        expected = [
            {"email_index": 0, "classification": "Not job-hunt related"},
            {"email_index": 1, "classification": "Confirmation"},
            {"email_index": 2, "classification": "Invitation"},
            {"email_index": 3, "classification": "Rejection"},
        ]

        self.assertEqual(results, expected)

    def test_train_multiclassifier_pipeline_invalid_classifier(self):
        """Method tests if the invalid classifier option triggers ClassifierOptionError route in train_multiclassifier_pipeline method."""

        app = EmailClassifierApp()

        # Mock CSV loading
        mock_df = pd.DataFrame({
            "email_text": ["hello"],
            "email_type": ["Invitation"],
        })
        app.load_data_csv = MagicMock(return_value=mock_df)

        # Return False for classifier option check → triggers exception
        app.classifier_option_check = MagicMock(return_value=False)

        # Track print output but remove noise
        with patch("builtins.print"):
            result = app.train_multiclassifier_pipeline(
                path="fake/path.csv",
                classifier_option="INVALID",
            )

        # Should return None
        self.assertIsNone(result)

        # The multiclassifier model must be set to None
        self.assertIsNone(app.multiclassifier_model)

    def test_train_multiclassifier_pipeline_success(self):
        """Method tests the success route of train_multiclassifier_pipeline method."""

        app = EmailClassifierApp()

        # --------------------------
        # Mock CSV loading
        # --------------------------
        mock_df = pd.DataFrame({
            "email_text": ["hello", "test"],
            "email_type": ["Invitation", "Rejection"],
        })

        app.load_data_csv = MagicMock(return_value=mock_df)

        # --------------------------
        # Mock classifier validation
        # --------------------------
        app.classifier_option_check = MagicMock(return_value=True)

        # --------------------------
        # Mock CLASSIFIERS dict so the classifier setter is a mock
        # --------------------------
        mock_setter = MagicMock()
        app.CLASSIFIERS = {"MultinomialNB": mock_setter}

        # --------------------------
        # Mock classifier object + its getter
        # --------------------------
        app.classifier = MagicMock()
        app.classifier.get_classifier.return_value = "mock_clf"

        # --------------------------
        # Mock model and builder
        # --------------------------
        mock_model = MagicMock()
        app.set_multiclassifier_model_clf = MagicMock(return_value=mock_model)

        # Run method
        app.train_multiclassifier_pipeline(
            path="fake/path.csv",
            classifier_option="MultinomialNB",
            column_name_train="email_type",
            column_name_main="email_text",
        )

        # --------------------------
        # Assertions
        # --------------------------

        # CSV loaded
        app.load_data_csv.assert_called_once_with("fake/path.csv")

        # classifier option check was used
        app.classifier_option_check.assert_called_once_with(
            "MultinomialNB",
            app.CLASSIFIERS
        )

        # Correct classifier setter was called
        mock_setter.assert_called_once()

        # Model was created using the classifier
        app.set_multiclassifier_model_clf.assert_called_once_with("mock_clf")

        # Pipeline procedures
        mock_model.build_pipeline.assert_called_once()
        mock_model.set_X.assert_called_once_with(mock_df["email_text"])
        mock_model.set_y.assert_called_once_with(mock_df["email_type"])
        mock_model.train.assert_called_once()

    def test_view_multiclassifier_accuracy_model_missing(self):
        """Method tests the view_multiclassifier_accuracy method when the multiclassifier model is missing."""

        app = EmailClassifierApp()
        app.multiclassifier_model = None  # Force missing model

        with patch("builtins.print") as mock_print:
            result = app.view_multiclassifier_accuracy()

        # Should return None
        self.assertIsNone(result)

        # Should print error message + code
        mock_print.assert_any_call("Model has not been initialized.")  # From ModelNotFound()
        mock_print.assert_any_call("Error code: MODEL_NOT_FOUND_003")  # Expected error code

    def test_view_multiclassifier_accuracy_success(self):
        """Method tests the view_multiclassifier_accuracy method success route."""

        app = EmailClassifierApp()

        # Mock model
        mock_model = MagicMock()
        mock_model.count_accuracy.return_value = (0.85, "classification report here")

        app.multiclassifier_model = mock_model

        with patch("builtins.print") as mock_print:
            result = app.view_multiclassifier_accuracy()

        # Method does not return anything specific
        self.assertIsNone(result)

        # Ensure count_accuracy() was called
        mock_model.count_accuracy.assert_called_once()

        # Ensure prints occurred in correct order
        mock_print.assert_any_call("Multiclassifier accuracy: ")
        mock_print.assert_any_call(0.85)
        mock_print.assert_any_call("classification report here")

    def test_predict_multiclassifier_model_not_found(self):
        """Method tests the predict_with_multiclassifier exception route when model is None."""

        app = EmailClassifierApp()
        app.multiclassifier_model = None

        with patch("builtins.print"):
            result = app.predict_with_multiclassifier("test email")

        self.assertIsNone(result)

    def test_predict_multiclassifier_invalid_input_type(self):
        """Method tests the predict_with_multiclassifier exception route when input is invalid."""

        app = EmailClassifierApp()

        app.multiclassifier_model = MagicMock()

        with patch("builtins.print"):
            result = app.predict_with_multiclassifier(12345)

        self.assertIsNone(result)

    def test_predict_multiclassifier_invalid_list_element(self):
        """Method tests the predict_with_multiclassifier exception route when element of input list is invalid."""

        app = EmailClassifierApp()

        app.multiclassifier_model = MagicMock()

        with patch("builtins.print"):
            result = app.predict_with_multiclassifier(["valid email", 999])

        self.assertIsNone(result)

    def test_predict_multiclassifier_success(self):
        """Method tests the predict_with_multiclassifier success route."""

        app = EmailClassifierApp()

        # --------------------------
        # Mock model and pipeline
        # --------------------------
        mock_model = MagicMock()
        mock_pipeline = MagicMock()
        mock_model.pipeline = mock_pipeline

        # Simulate predictions
        mock_pipeline.predict.side_effect = [
            ["Invitation"],
            ["Other"],
            ["Confirmation"],
        ]

        app.multiclassifier_model = mock_model

        emails = [
            "email one",
            "email two",
            "email three",
        ]

        with patch("builtins.print"):
            results = app.predict_with_multiclassifier(emails)

        expected = [
            {"email_index": 0, "classification": "Invitation"},
            {"email_index": 1, "classification": "Other"},
            {"email_index": 2, "classification": "Confirmation"},
        ]

        self.assertEqual(results, expected)

        # Ensure pipeline.predict was called for each email
        self.assertEqual(mock_pipeline.predict.call_count, 3)

    def test_train_multiclassifier_all_classifier_options(self):
        """Method that tests all possible supported classifiers usage for multiclassifier training."""

        app = EmailClassifierApp()

        # Mock dataset
        mock_df = pd.DataFrame({
            "email_text": ["a", "b"],
            "email_type": ["Invitation", "Rejection"]
        })

        app.load_data_csv = MagicMock(return_value=mock_df)
        app.classifier_option_check = MagicMock(return_value=True)

        # Mock model
        mock_model = MagicMock()
        app.set_multiclassifier_model_clf = MagicMock(return_value=mock_model)

        # Mock classifier instance
        app.classifier = MagicMock()
        app.classifier.get_classifier.return_value = "mock_clf"

        # Prepare fake CLASSIFIERS map
        fake_classifiers = {}
        for key in app.CLASSIFIERS.keys():
            fake_classifiers[key] = MagicMock(name=f"set_{key}")

        app.CLASSIFIERS = fake_classifiers

        # Skip special classifiers here
        excluded = {"VotingClassifier", "StackingClassifier"}

        for clf_name in app.CLASSIFIERS:
            if clf_name in excluded:
                continue

            with self.subTest(classifier=clf_name):
                app.train_multiclassifier_pipeline(
                    path="fake.csv",
                    classifier_option=clf_name
                )

                # correct setter called
                app.CLASSIFIERS[clf_name].assert_called_once()

                # model pipeline executed
                mock_model.build_pipeline.assert_called()
                mock_model.set_X.assert_called_with(mock_df["email_text"])
                mock_model.set_y.assert_called_with(mock_df["email_type"])
                mock_model.train.assert_called()

                # reset mocks for next iteration
                app.CLASSIFIERS[clf_name].reset_mock()
                mock_model.reset_mock()

    def test_train_multiclassifier_voting_classifier(self):
        """Method that tests StackingClassifier getting set as multiclassifier model."""

        app = EmailClassifierApp()

        mock_df = pd.DataFrame({
            "email_text": ["x"],
            "email_type": ["Other"]
        })

        app.load_data_csv = MagicMock(return_value=mock_df)
        app.classifier_option_check = MagicMock(return_value=True)

        app.classifier = MagicMock()
        app.classifier.get_classifier.return_value = "mock_clf"

        app.set_voting_classifier_parameters = MagicMock()

        mock_model = MagicMock()
        app.set_multiclassifier_model_clf = MagicMock(return_value=mock_model)

        app.CLASSIFIERS = {
            "VotingClassifier": MagicMock()
        }

        app.train_multiclassifier_pipeline(
            classifier_option="VotingClassifier",
            estimator_1="A",
            estimator_2="B",
            estimator_3="C",
            voting_option="soft"
        )

        app.CLASSIFIERS["VotingClassifier"].assert_called_once()
        app.set_voting_classifier_parameters.assert_called_once_with(
            "A", "B", "C", "soft"
        )
        mock_model.train.assert_called_once()

    def test_train_multiclassifier_stacking_classifier(self):
        """Method that tests StackingClassifier getting set as multiclassifier model."""

        app = EmailClassifierApp()

        mock_df = pd.DataFrame({
            "email_text": ["x"],
            "email_type": ["Confirmation"]
        })

        app.load_data_csv = MagicMock(return_value=mock_df)
        app.classifier_option_check = MagicMock(return_value=True)

        app.classifier = MagicMock()
        app.classifier.get_classifier.return_value = "mock_clf"

        app.set_stacking_classifier_estimators = MagicMock()

        mock_model = MagicMock()
        app.set_multiclassifier_model_clf = MagicMock(return_value=mock_model)

        app.CLASSIFIERS = {
            "StackingClassifier": MagicMock()
        }

        app.train_multiclassifier_pipeline(
            classifier_option="StackingClassifier",
            estimator_1="A",
            estimator_2="B",
            estimator_3="C"
        )

        app.CLASSIFIERS["StackingClassifier"].assert_called_once()
        app.set_stacking_classifier_estimators.assert_called_once_with(
            "A", "B", "C"
        )
        mock_model.train.assert_called_once()

    def test_train_3_stage_pipelines_all_supported_classifiers(self):
        """Method that tests all possible supported classifiers usage for 3 stage pipelines training."""

        app = EmailClassifierApp()

        # --------------------------
        # Mock dataset
        # --------------------------
        mock_df = pd.DataFrame({
            "email_text": ["a", "b"],
            "related_to_jobhunt": [True, False],
            "is_confirmation": [False, True],
            "is_invitation": [True, False],
        })

        app.load_data_csv = MagicMock(return_value=mock_df)
        app.classifier_option_check = MagicMock(return_value=True)

        # --------------------------
        # Mock classifier instance
        # --------------------------
        app.classifier = MagicMock()
        app.classifier.get_classifier.return_value = "mock_clf"

        # --------------------------
        # Mock models
        # --------------------------
        model1 = MagicMock()
        model2 = MagicMock()
        model3 = MagicMock()

        app.set_model1_clf = MagicMock(return_value=model1)
        app.set_model2_clf = MagicMock(return_value=model2)
        app.set_model3_clf = MagicMock(return_value=model3)

        # --------------------------
        # Prepare CLASSIFIERS map
        # --------------------------
        fake_classifiers = {}
        for name in app.CLASSIFIERS.keys():
            fake_classifiers[name] = MagicMock(name=f"set_{name}")

        app.CLASSIFIERS = fake_classifiers

        # --------------------------
        # Exclude unsupported classifiers
        # --------------------------
        unsupported = {"VotingClassifier", "StackingClassifier"}
        supported = [c for c in app.CLASSIFIERS if c not in unsupported]

        # --------------------------
        # Rotate classifiers across stages
        # --------------------------
        for i in range(0, len(supported), 3):
            clf1 = supported[i]
            clf2 = supported[(i + 1) % len(supported)]
            clf3 = supported[(i + 2) % len(supported)]

            with self.subTest(stage1=clf1, stage2=clf2, stage3=clf3):

                app.train_3_stage_pipelines(
                    path1="fake1.csv",
                    path2="fake2.csv",
                    path3="fake3.csv",
                    classifier_option_1=clf1,
                    classifier_option_2=clf2,
                    classifier_option_3=clf3,
                )

                # Ensure correct classifier setters were called
                app.CLASSIFIERS[clf1].assert_called_once()
                app.CLASSIFIERS[clf2].assert_called_once()
                app.CLASSIFIERS[clf3].assert_called_once()

                # Ensure training executed
                model1.train.assert_called()
                model2.train.assert_called()
                model3.train.assert_called()

                # Reset mocks for next rotation
                for clf in (clf1, clf2, clf3):
                    app.CLASSIFIERS[clf].reset_mock()

                model1.reset_mock()
                model2.reset_mock()
                model3.reset_mock()

    def test_train_3_stage_pipeline_unsupported_classifiers_fallback(self):
        """Method that tests if Voting and Stacking classifiers are reset to MultinomialNB in 3 stage pipeline setup."""

        app = EmailClassifierApp()

        app.load_data_csv = MagicMock(return_value=pd.DataFrame({
            "email_text": ["x"],
            "related_to_jobhunt": [True],
            "is_confirmation": [False],
            "is_invitation": [True],
        }))

        app.classifier_option_check = MagicMock(return_value=True)
        app.classifier = MagicMock()
        app.classifier.get_classifier.return_value = "mock_clf"

        app.CLASSIFIERS = {
            "MultinomialNB": MagicMock(),
            "VotingClassifier": MagicMock(),
            "StackingClassifier": MagicMock(),
        }

        app.set_model1_clf = MagicMock(return_value=MagicMock())
        app.set_model2_clf = MagicMock(return_value=MagicMock())
        app.set_model3_clf = MagicMock(return_value=MagicMock())

        with patch("builtins.print"):
            app.train_3_stage_pipelines(
                classifier_option_1="VotingClassifier",
                classifier_option_2="StackingClassifier",
                classifier_option_3="VotingClassifier",
            )

        app.CLASSIFIERS["MultinomialNB"].assert_called()

    def test_set_voting_classifier_parameters_invalid_voting_option(self):
        """Method that tests set_voting_classifier_parameters method exception route when voting option is invalid."""
        app = EmailClassifierApp()

        app.classifier = MagicMock()
        app.classifier_option_check = MagicMock(return_value=True)

        app.classifier.ESTIMATORS_AND_CLASSIFIERS = {
            "MultinomialNB": MagicMock()
        }

        result = app.set_voting_classifier_parameters(
            estimator_1="MultinomialNB",
            estimator_2="MultinomialNB",
            estimator_3="MultinomialNB",
            voting_option="invalid_option",
        )

        self.assertIsNone(result)

        app.classifier.set_voting_hard.assert_not_called()
        app.classifier.set_voting_soft.assert_not_called()
        app.classifier.set_clf_vtc.assert_not_called()

    def test_set_voting_classifier_parameters_invalid_estimator_1(self):
        """Method that tests set_voting_classifier_parameters method exception route when estimator_1 option is invalid."""
        app = EmailClassifierApp()

        app.classifier = MagicMock()
        app.classifier_option_check = MagicMock(
            side_effect=[True, False, True]
        )

        app.classifier.ESTIMATORS_AND_CLASSIFIERS = {
            "MultinomialNB": MagicMock()
        }

        app.set_voting_classifier_parameters(
            estimator_1="InvalidEstimator",
            estimator_2="MultinomialNB",
            estimator_3="MultinomialNB",
            voting_option="hard",
        )

        # Voting mode should be set
        app.classifier.set_voting_hard.assert_called_once()

        # Estimators should NOT be set
        app.classifier.set_vc_clf_1.assert_not_called()
        app.classifier.set_vc_clf_2.assert_not_called()
        app.classifier.set_vc_clf_3.assert_not_called()
        app.classifier.set_clf_vtc.assert_not_called()

    def test_set_voting_classifier_parameters_invalid_estimator_2(self):
        """Method that tests set_voting_classifier_parameters method exception route when estimator_2 option is invalid."""
        app = EmailClassifierApp()

        app.classifier = MagicMock()
        app.classifier_option_check = MagicMock(
            side_effect=[True, False, True]
        )

        app.classifier.ESTIMATORS_AND_CLASSIFIERS = {
            "MultinomialNB": MagicMock()
        }

        app.set_voting_classifier_parameters(
            estimator_1="MultinomialNB",
            estimator_2="InvalidEstimator",
            estimator_3="MultinomialNB",
            voting_option="hard",
        )

        # Voting mode should be set
        app.classifier.set_voting_hard.assert_called_once()

        # Estimators should NOT be set
        app.classifier.set_vc_clf_1.assert_not_called()
        app.classifier.set_vc_clf_2.assert_not_called()
        app.classifier.set_vc_clf_3.assert_not_called()
        app.classifier.set_clf_vtc.assert_not_called()

    def test_set_voting_classifier_parameters_invalid_estimator_3(self):
        """Method that tests set_voting_classifier_parameters method exception route when estimator_3 option is invalid."""
        app = EmailClassifierApp()

        app.classifier = MagicMock()
        app.classifier_option_check = MagicMock(
            side_effect=[True, False, True] 
        )

        app.classifier.ESTIMATORS_AND_CLASSIFIERS = {
            "MultinomialNB": MagicMock()
        }

        app.set_voting_classifier_parameters(
            estimator_1="MultinomialNB",
            estimator_2="MultinomialNB",
            estimator_3="InvalidEstimator",
            voting_option="hard",
        )

        # Voting mode should be set
        app.classifier.set_voting_hard.assert_called_once()

        # Estimators should NOT be set
        app.classifier.set_vc_clf_1.assert_not_called()
        app.classifier.set_vc_clf_2.assert_not_called()
        app.classifier.set_vc_clf_3.assert_not_called()
        app.classifier.set_clf_vtc.assert_not_called()

    def test_set_voting_classifier_parameters_invalid_estimator_1_2(self):
        """Method that tests set_voting_classifier_parameters method exception route when estimator_1 and estimator_2 options are invalid."""
        app = EmailClassifierApp()

        app.classifier = MagicMock()
        app.classifier_option_check = MagicMock(
            side_effect=[True, False, True]
        )

        app.classifier.ESTIMATORS_AND_CLASSIFIERS = {
            "MultinomialNB": MagicMock()
        }

        app.set_voting_classifier_parameters(
            estimator_1="InvalidEstimator",
            estimator_2="InvalidEstimator",
            estimator_3="MultinomialNB",
            voting_option="hard",
        )

        # Voting mode should be set
        app.classifier.set_voting_hard.assert_called_once()

        # Estimators should NOT be set
        app.classifier.set_vc_clf_1.assert_not_called()
        app.classifier.set_vc_clf_2.assert_not_called()
        app.classifier.set_vc_clf_3.assert_not_called()
        app.classifier.set_clf_vtc.assert_not_called()

    def test_set_voting_classifier_parameters_invalid_estimator_1_3(self):
        """Method that tests set_voting_classifier_parameters method exception route when estimator_1 and estimator_3 options are invalid."""
        app = EmailClassifierApp()

        app.classifier = MagicMock()
        app.classifier_option_check = MagicMock(
            side_effect=[True, False, True]
        )

        app.classifier.ESTIMATORS_AND_CLASSIFIERS = {
            "MultinomialNB": MagicMock()
        }

        app.set_voting_classifier_parameters(
            estimator_1="InvalidEstimator",
            estimator_2="MultinomialNB",
            estimator_3="InvalidEstimator",
            voting_option="hard",
        )

        # Voting mode should be set
        app.classifier.set_voting_hard.assert_called_once()

        # Estimators should NOT be set
        app.classifier.set_vc_clf_1.assert_not_called()
        app.classifier.set_vc_clf_2.assert_not_called()
        app.classifier.set_vc_clf_3.assert_not_called()
        app.classifier.set_clf_vtc.assert_not_called()

    def test_set_voting_classifier_parameters_invalid_estimator_2_3(self):
        """Method that tests set_voting_classifier_parameters method exception route when estimator_2 and estimator_3 options are invalid."""
        app = EmailClassifierApp()

        app.classifier = MagicMock()
        app.classifier_option_check = MagicMock(
            side_effect=[True, False, True]
        )

        app.classifier.ESTIMATORS_AND_CLASSIFIERS = {
            "MultinomialNB": MagicMock()
        }

        app.set_voting_classifier_parameters(
            estimator_1="MultinomialNB",
            estimator_2="InvalidEstimator",
            estimator_3="InvalidEstimator",
            voting_option="hard",
        )

        # Voting mode should be set
        app.classifier.set_voting_hard.assert_called_once()

        # Estimators should NOT be set
        app.classifier.set_vc_clf_1.assert_not_called()
        app.classifier.set_vc_clf_2.assert_not_called()
        app.classifier.set_vc_clf_3.assert_not_called()
        app.classifier.set_clf_vtc.assert_not_called()

    def test_set_voting_classifier_parameters_invalid_estimator_1_2_3_h(self):
        """Method that tests set_voting_classifier_parameters method exception route when estimator_1, estimator_2 and estimator_3 options are invalid."""
        app = EmailClassifierApp()

        app.classifier = MagicMock()
        app.classifier_option_check = MagicMock(
            side_effect=[True, False, True]
        )

        app.classifier.ESTIMATORS_AND_CLASSIFIERS = {
            "MultinomialNB": MagicMock()
        }

        app.set_voting_classifier_parameters(
            estimator_1="InvalidEstimator",
            estimator_2="InvalidEstimator",
            estimator_3="InvalidEstimator",
            voting_option="hard",
        )

        # Voting mode should be set
        app.classifier.set_voting_hard.assert_called_once()

        # Estimators should NOT be set
        app.classifier.set_vc_clf_1.assert_not_called()
        app.classifier.set_vc_clf_2.assert_not_called()
        app.classifier.set_vc_clf_3.assert_not_called()
        app.classifier.set_clf_vtc.assert_not_called()

    def test_set_voting_classifier_parameters_invalid_estimator_1_2_3_s(self):
        """Method that tests set_voting_classifier_parameters method exception route when estimator_1, estimator_2 and estimator_3 options are invalid, and voting option is soft."""
        app = EmailClassifierApp()

        app.classifier = MagicMock()
        app.classifier_option_check = MagicMock(
            side_effect=[True, False, True]
        )

        app.classifier.ESTIMATORS_AND_CLASSIFIERS = {
            "MultinomialNB": MagicMock()
        }

        app.set_voting_classifier_parameters(
            estimator_1="InvalidEstimator",
            estimator_2="InvalidEstimator",
            estimator_3="InvalidEstimator",
            voting_option="soft",
        )

        # Voting mode should be set
        app.classifier.set_voting_soft.assert_called_once()

        # Estimators should NOT be set
        app.classifier.set_vc_clf_1.assert_not_called()
        app.classifier.set_vc_clf_2.assert_not_called()
        app.classifier.set_vc_clf_3.assert_not_called()
        app.classifier.set_clf_vtc.assert_not_called()

    def test_set_voting_classifier_parameters_invalid_estimator_1_2_3_inv(self):
        """Method that tests set_voting_classifier_parameters method exception route when estimator_1, estimator_2 and estimator_3 options are invalid, and voting option is invalid as well."""
        app = EmailClassifierApp()

        app.classifier = MagicMock()
        app.classifier_option_check = MagicMock(
            side_effect=[True, False, True]
        )

        app.classifier.ESTIMATORS_AND_CLASSIFIERS = {
            "MultinomialNB": MagicMock()
        }

        app.set_voting_classifier_parameters(
            estimator_1="InvalidEstimator",
            estimator_2="InvalidEstimator",
            estimator_3="InvalidEstimator",
            voting_option="InvalidOption",
        )

        app.classifier.set_voting_hard.assert_not_called()
        app.classifier.set_voting_soft.assert_not_called()

        # Estimators should NOT be set
        app.classifier.set_vc_clf_1.assert_not_called()
        app.classifier.set_vc_clf_2.assert_not_called()
        app.classifier.set_vc_clf_3.assert_not_called()
        app.classifier.set_clf_vtc.assert_not_called()

    def test_set_voting_classifier_parameters_success_default_estimators(self):
        """Method that tests set_voting_classifier_parameters method's success route with default estimators."""
        app = EmailClassifierApp()

        # --------------------------
        # Mock classifier
        # --------------------------
        app.classifier = MagicMock()

        # --------------------------
        # Mock estimator registry
        # --------------------------
        mock_estimator = MagicMock()
        app.classifier.ESTIMATORS_AND_CLASSIFIERS = {
            "MultinomialNB": mock_estimator
        }

        # --------------------------
        # All estimator checks pass
        # --------------------------
        app.classifier_option_check = MagicMock(return_value=True)

        # --------------------------
        # Run method
        # --------------------------
        app.set_voting_classifier_parameters(
            estimator_1="MultinomialNB",
            estimator_2="MultinomialNB",
            estimator_3="MultinomialNB",
            voting_option="hard",
        )

        # --------------------------
        # Assertions
        # --------------------------

        # Voting mode set
        app.classifier.set_voting_hard.assert_called_once()

        # Estimators created
        self.assertEqual(mock_estimator.call_count, 3)

        # Estimators assigned
        app.classifier.set_vc_clf_1.assert_called_once()
        app.classifier.set_vc_clf_2.assert_called_once()
        app.classifier.set_vc_clf_3.assert_called_once()

        # Final classifier built
        app.classifier.set_clf_vtc.assert_called_once()

    def test_set_voting_classifier_parameters_all_estimators(self):
        """Method that tests set_voting_classifier_parameters method's success route with all possible estimators."""
        app = EmailClassifierApp()

        # --------------------------
        # Mock classifier
        # --------------------------
        app.classifier = MagicMock()

        # --------------------------
        # Use real estimator names but mock constructors
        # --------------------------
        estimator_names = [
            "MultinomialNB",
            "LogisticRegression",
            "ComplementNB",
            "BernoulliNB",
            "SGDClassifier",
            "RidgeClassifier",
            "RandomForestClassifier",
            "GradientBoostingClassifier",
            "AdaBoostClassifier",
            "LinearSVC",
            "SVC",
            "KNeighborsClassifier",
            "DecisionTreeClassifier",
            "ExtraTreeClassifier",
        ]

        # Build mocked registry
        app.classifier.ESTIMATORS_AND_CLASSIFIERS = {
            name: MagicMock(name=f"{name}_ctor") for name in estimator_names
        }

        # All estimator options valid
        app.classifier_option_check = MagicMock(return_value=True)

        # --------------------------
        # Test both voting options
        # --------------------------
        for voting_option in ("hard", "soft"):
            for estimator in estimator_names:

                # Reset mocks between runs
                app.classifier.reset_mock()
                for ctor in app.classifier.ESTIMATORS_AND_CLASSIFIERS.values():
                    ctor.reset_mock()

                app.set_voting_classifier_parameters(
                    estimator_1=estimator,
                    estimator_2=estimator,
                    estimator_3=estimator,
                    voting_option=voting_option,
                )

                # --------------------------
                # Assertions
                # --------------------------

                # Correct voting mode
                if voting_option == "hard":
                    app.classifier.set_voting_hard.assert_called_once()
                    app.classifier.set_voting_soft.assert_not_called()
                else:
                    app.classifier.set_voting_soft.assert_called_once()
                    app.classifier.set_voting_hard.assert_not_called()

                # Estimator constructor called 3 times
                ctor = app.classifier.ESTIMATORS_AND_CLASSIFIERS[estimator]
                self.assertEqual(ctor.call_count, 3)

                # Estimators assigned
                app.classifier.set_vc_clf_1.assert_called_once()
                app.classifier.set_vc_clf_2.assert_called_once()
                app.classifier.set_vc_clf_3.assert_called_once()

                # VotingClassifier finalized
                app.classifier.set_clf_vtc.assert_called_once()

    def test_set_stacking_classifier_estimators_all_none(self):
        """Method that tests set_stacking_classifier_estimators when all estimators are none."""
        app = EmailClassifierApp()
        app.classifier = MagicMock()

        result = app.set_stacking_classifier_estimators(
            estimator_1=None,
            estimator_2=None,
            estimator_3=None,
        )

        self.assertIsNone(result)

        # No classifier methods should be called
        app.classifier.set_sc_clf_1.assert_not_called()
        app.classifier.set_sc_clf_2.assert_not_called()
        app.classifier.set_sc_clf_3.assert_not_called()
        app.classifier.set_clf_stc.assert_not_called()

    def test_set_stacking_classifier_estimators_invalid_estimator_1(self):
        """Method that tests set_stacking_classifier_estimators exception route when estimator_1 is invalid."""
        app = EmailClassifierApp()
        app.classifier = MagicMock()

        # Registry with only one valid option
        app.classifier.ESTIMATORS_AND_CLASSIFIERS = {
            "MultinomialNB": MagicMock()
        }

        # One estimator invalid
        app.classifier_option_check = MagicMock(
            side_effect=[True, False, True]
        )

        result = app.set_stacking_classifier_estimators(
            estimator_1="InvalidEstimator",
            estimator_2="MultinomialNB",
            estimator_3="MultinomialNB",
        )

        self.assertIsNone(result)

        # No classifier building should happen
        app.classifier.set_sc_clf_1.assert_not_called()
        app.classifier.set_sc_clf_2.assert_not_called()
        app.classifier.set_sc_clf_3.assert_not_called()
        app.classifier.set_clf_stc.assert_not_called()

    def test_set_stacking_classifier_estimators_invalid_estimator_2(self):
        """Method that tests set_stacking_classifier_estimators exception route when estimator_2 is invalid."""
        app = EmailClassifierApp()
        app.classifier = MagicMock()

        # Registry with only one valid option
        app.classifier.ESTIMATORS_AND_CLASSIFIERS = {
            "MultinomialNB": MagicMock()
        }

        # One estimator invalid
        app.classifier_option_check = MagicMock(
            side_effect=[True, False, True]
        )

        result = app.set_stacking_classifier_estimators(
            estimator_1="MultinomialNB",
            estimator_2="InvalidEstimator",
            estimator_3="MultinomialNB",
        )

        self.assertIsNone(result)

        # No classifier building should happen
        app.classifier.set_sc_clf_1.assert_not_called()
        app.classifier.set_sc_clf_2.assert_not_called()
        app.classifier.set_sc_clf_3.assert_not_called()
        app.classifier.set_clf_stc.assert_not_called()

    def test_set_stacking_classifier_estimators_invalid_estimator_3(self):
        """Method that tests set_stacking_classifier_estimators exception route when estimator_3 is invalid."""
        app = EmailClassifierApp()
        app.classifier = MagicMock()

        # Registry with only one valid option
        app.classifier.ESTIMATORS_AND_CLASSIFIERS = {
            "MultinomialNB": MagicMock()
        }

        # One estimator invalid
        app.classifier_option_check = MagicMock(
            side_effect=[True, False, True]
        )

        result = app.set_stacking_classifier_estimators(
            estimator_1="MultinomialNB",
            estimator_2="MultinomialNB",
            estimator_3="InvalidEstimator",
        )

        self.assertIsNone(result)

        # No classifier building should happen
        app.classifier.set_sc_clf_1.assert_not_called()
        app.classifier.set_sc_clf_2.assert_not_called()
        app.classifier.set_sc_clf_3.assert_not_called()
        app.classifier.set_clf_stc.assert_not_called()

    def test_set_stacking_classifier_estimators_success_default(self):
        app = EmailClassifierApp()
        app.classifier = MagicMock()

        # Mock estimator constructor
        mock_estimator_ctor = MagicMock(name="MultinomialNB_ctor")

        app.classifier.ESTIMATORS_AND_CLASSIFIERS = {
            "MultinomialNB": mock_estimator_ctor
        }

        # All estimator checks pass
        app.classifier_option_check = MagicMock(return_value=True)

        app.set_stacking_classifier_estimators(
            estimator_1="MultinomialNB",
            estimator_2="MultinomialNB",
            estimator_3="MultinomialNB",
        )

        # Constructor called 3 times
        self.assertEqual(mock_estimator_ctor.call_count, 3)

        # Estimators assigned
        app.classifier.set_sc_clf_1.assert_called_once()
        app.classifier.set_sc_clf_2.assert_called_once()
        app.classifier.set_sc_clf_3.assert_called_once()

        # Final StackingClassifier creation
        app.classifier.set_clf_stc.assert_called_once()

    def test_set_stacking_classifier_estimators_all_estimators(self):
        """Method that tests set_stacking_classifier_estimators with all supported estimators."""
        app = EmailClassifierApp()
        app.classifier = MagicMock()

        estimator_names = [
            "MultinomialNB",
            "LogisticRegression",
            "ComplementNB",
            "BernoulliNB",
            "SGDClassifier",
            "RidgeClassifier",
            "RandomForestClassifier",
            "GradientBoostingClassifier",
            "AdaBoostClassifier",
            "LinearSVC",
            "SVC",
            "KNeighborsClassifier",
            "DecisionTreeClassifier",
            "ExtraTreeClassifier",
        ]


        # Mock estimator constructors
        estimators_dict = {}
        for name in estimator_names:
            estimators_dict[name] = MagicMock(name=f"{name}_ctor")

        app.classifier.ESTIMATORS_AND_CLASSIFIERS = estimators_dict

        # All estimator checks pass
        app.classifier_option_check = MagicMock(return_value=True)

        # --------------------------
        # Iterate through estimators
        # --------------------------
        for estimator_name in estimator_names:
            # Reset mocks before each iteration
            app.classifier.reset_mock()
            for ctor in estimators_dict.values():
                ctor.reset_mock()

            app.set_stacking_classifier_estimators(
                estimator_1=estimator_name,
                estimator_2=estimator_name,
                estimator_3=estimator_name,
            )

            # Constructor should be called 3 times
            self.assertEqual(
                estimators_dict[estimator_name].call_count,
                3,
                msg=f"Estimator {estimator_name} was not instantiated 3 times",
            )

            # Estimators assigned
            app.classifier.set_sc_clf_1.assert_called_once()
            app.classifier.set_sc_clf_2.assert_called_once()
            app.classifier.set_sc_clf_3.assert_called_once()

            # Final classifier creation
            app.classifier.set_clf_stc.assert_called_once()

if __name__ == "__main__":
    unittest.main()
