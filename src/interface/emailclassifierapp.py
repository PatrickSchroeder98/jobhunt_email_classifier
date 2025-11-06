from src.classifiers.classifier import Classifier
from src.data_module.dataloader import DataLoader
from src.exceptions.exceptions import (
    PathError,
    ClassifierOptionError,
    ModelNotFound,
    InputDataError,
    VotingOptionForVotingClassifierError,
    EstimatorOptionError,
    VotingClassifierNotSupported,
)
from src.models.model import Model


class EmailClassifierApp:
    """Class that is the interface for EmailClassifier"""

    def __init__(self):
        """Constructor initializes data_loader and classifier. Model is None at this point."""
        self.data_loader = DataLoader()
        self.classifier = Classifier()
        self.model1 = None
        self.model2 = None
        self.model3 = None
        self.multiclassifier_model = None
        self.CLASSIFIERS = {
            "MultinomialNB": self.classifier.set_clf_nb,
            "LogisticRegression": self.classifier.set_clf_lr,
            "ComplementNB": self.classifier.set_clf_cnb,
            "BernoulliNB": self.classifier.set_clf_bnb,
            "SGDClassifier": self.classifier.set_clf_sgd,
            "RidgeClassifier": self.classifier.set_clf_rdg,
            "RandomForestClassifier": self.classifier.set_clf_rfc,
            "GradientBoostingClassifier": self.classifier.set_clf_gbc,
            "AdaBoostClassifier": self.classifier.set_clf_abc,
            "LinearSVC": self.classifier.set_clf_lsv,
            "SVC": self.classifier.set_clf_svc,
            "KNeighborsClassifier": self.classifier.set_clf_knn,
            "VotingClassifier": self.classifier.set_clf_vtc,
        }

    def set_model1_clf(self, clf):
        """Method sets model object with provided classifier and returns it."""
        self.model1 = Model(clf)
        return self.model1

    def set_model2_clf(self, clf):
        """Method sets model object with provided classifier and returns it."""
        self.model2 = Model(clf)
        return self.model2

    def set_model3_clf(self, clf):
        """Method sets model object with provided classifier and returns it."""
        self.model3 = Model(clf)
        return self.model3

    def set_multiclassifier_model_clf(self, clf):
        """Method sets model object with provided classifier and returns it."""
        self.multiclassifier_model = Model(clf)
        return self.multiclassifier_model

    def load_data_csv(self, path):
        """Method loads data from csv file. If it fails, an exception is raised."""
        try:
            self.data_loader.set_path(path)
            df = self.data_loader.load_data_csv()
            return df
        except PathError as e:
            print(e.get_message())
            print("Error code: " + e.get_code())
            return None

    def classifier_option_check(self, option, constant):
        """Method that checks if provided classifier option is valid."""
        return False if option not in constant.keys() else True

    def train_3_stage_pipelines(
        self,
        path1="../data/training_emails_stage_1.csv",
        path2="../data/training_emails_stage_2.csv",
        path3="../data/training_emails_stage_3.csv",
        classifier_option_1="MultinomialNB",
        classifier_option_2="MultinomialNB",
        classifier_option_3="MultinomialNB",
        column_name_1="related_to_jobhunt",
        column_name_2="is_confirmation",
        column_name_3="is_invitation",
        column_name_main="email_text",
    ):
        """Method trains 3 pipelines with the usage of data given in argument. Default option is the data from 'data' folder"""

        def train_pipeline(path, classifier_option, setup, column_name_train):
            """Helper function for training pipeline."""
            try:
                df = self.load_data_csv(path)
                if not self.classifier_option_check(classifier_option, self.CLASSIFIERS):
                    raise ClassifierOptionError
                else:
                    try:
                        if classifier_option == "VotingClassifier":
                            raise VotingClassifierNotSupported
                    except VotingClassifierNotSupported as e:
                        print(e.get_message())
                        print("Error code: " + e.get_code())
                        print("Classifier reset to default MultinomialNB")
                        classifier_option = "MultinomialNB"
                    finally:
                        # uses the key of dictionary and calls the corresponding method
                        self.CLASSIFIERS[classifier_option]()
            except ClassifierOptionError as e:
                print(e.get_message())
                print("Error code: " + e.get_code())
                self.model1 = None
                self.model2 = None
                self.model3 = None
                return None
            else:
                model = setup(self.classifier.get_classifier())
                model.build_pipeline()
                model.set_X(df[column_name_main])
                model.set_y(df[column_name_train])
                model.train()

        train_pipeline(path1, classifier_option_1, self.set_model1_clf, column_name_1)
        train_pipeline(path2, classifier_option_2, self.set_model2_clf, column_name_2)
        train_pipeline(path3, classifier_option_3, self.set_model3_clf, column_name_3)

    def view_3_stage_pipelines_accuracy(self):
        """Method displays the accuracy of the 3 pipelines."""
        try:
            if self.model1 is None or self.model2 is None or self.model3 is None:
                raise ModelNotFound()
        except ModelNotFound as e:
            print(e.get_message())
            print("Error code: " + e.get_code())
            return None
        else:
            def view(stage, model):
                """Helper function for viewing 3 stage pipelines accuracy."""
                print(stage + " stage accuracy: ")
                accuracy = model.count_accuracy()
                print(accuracy[0])
                print(accuracy[1])

            stages = ["1st", "2nd", "3rd"]
            models = [self.model1, self.model2, self.model3]

            for s, m in zip(stages, models):
                view(s, m)

    def classify_emails_3_stage_pipelines(self, emails):
        """Classify one or multiple emails through 3-stage pipelines."""
        try:
            if self.model1 is None or self.model2 is None or self.model3 is None:
                raise ModelNotFound()
        except ModelNotFound as e:
            print(e.get_message())
            print("Error code: " + e.get_code())
            return None
        else:

            if isinstance(emails, str):
                emails = [emails]
            try:
                if not isinstance(emails, list):
                    raise InputDataError()

                for email in emails:
                    if not isinstance(email, str):
                        raise InputDataError()
            except InputDataError as e:
                print(e.get_message())
                print("Error code: " + e.get_code())
                return None

            results = []

            for i, text in enumerate(emails):
                # --- Stage 1: Is it jobhunt-related? ---
                prediction_stage_1 = self.model1.pipeline.predict([text])[0]
                if not bool(prediction_stage_1):
                    results.append(
                        {"email_index": i, "classification": "Not job-hunt related"}
                    )
                    continue

                # --- Stage 2: Confirmation or next step? ---
                prediction_stage2 = self.model2.pipeline.predict([text])[0]
                if bool(prediction_stage2):
                    results.append({"email_index": i, "classification": "Confirmation"})
                    continue

                # --- Stage 3: Invitation or Rejection? ---
                prediction_stage3 = self.model3.pipeline.predict([text])[0]
                classification = (
                    "Invitation" if bool(prediction_stage3) else "Rejection"
                )

                results.append({"email_index": i, "classification": classification})

            return results

    def train_multiclassifier_pipeline(
        self,
        path="../data/training_emails_multiclassifier.csv",
        classifier_option="MultinomialNB",
        column_name_train="email_type",
        column_name_main="email_text",
        estimator_1 = None,
        estimator_2 = None,
        estimator_3 = None,
        voting_option= "hard"
    ):
        """Method trains a pipeline that utilizes multiclassification."""
        try:
            df = self.load_data_csv(path)
            if not self.classifier_option_check(classifier_option, self.CLASSIFIERS):
                raise ClassifierOptionError
            else:
                # uses the key of dictionary and calls the corresponding method
                self.CLASSIFIERS[classifier_option]()

                if classifier_option == "VotingClassifier":
                    self.set_voting_classifier_parameters(estimator_1, estimator_2, estimator_3, voting_option)
        except ClassifierOptionError as e:
            print(e.get_message())
            print("Error code: " + e.get_code())
            self.multiclassifier_model = None
            return None
        else:
            model = self.set_multiclassifier_model_clf(self.classifier.get_classifier())
            model.build_pipeline()
            model.set_X(df[column_name_main])
            model.set_y(df[column_name_train])
            model.train()

    def view_multiclassifier_accuracy(self):
        """Method displays the accuracy of the multiclassifier."""
        try:
            if self.multiclassifier_model is None:
                raise ModelNotFound()
        except ModelNotFound as e:
            print(e.get_message())
            print("Error code: " + e.get_code())
            return None
        else:
            print("Multiclassifier accuracy: ")
            accuracy = self.multiclassifier_model.count_accuracy()
            print(accuracy[0])
            print(accuracy[1])

    def predict_with_multiclassifier(self, emails):
        """Method predicts emails' type with multiclassifier."""

        try:
            if self.multiclassifier_model is None:
                raise ModelNotFound()
        except ModelNotFound as e:
            print(e.get_message())
            print("Error code: " + e.get_code())
            return None
        else:

            if isinstance(emails, str):
                emails = [emails]
            try:
                if not isinstance(emails, list):
                    raise InputDataError()

                for email in emails:
                    if not isinstance(email, str):
                        raise InputDataError()
            except InputDataError as e:
                print(e.get_message())
                print("Error code: " + e.get_code())
                return None

            results = []

            for i, text in enumerate(emails):
                prediction_stage_1 = self.multiclassifier_model.pipeline.predict([text])[0]
                results.append({"email_index": i, "classification": str(prediction_stage_1)})

            return results

    def set_voting_classifier_parameters(self, estimator_1, estimator_2, estimator_3, voting_option):
        """Method allows user to set estimators and voting option for VotingClassifier."""

        if estimator_1 is None and estimator_2 is None and estimator_3 is None:
            return None

        try:
            if voting_option == "hard":
                self.classifier.set_voting_hard()
            elif voting_option == "soft":
                self.classifier.set_voting_soft()
            else:
                raise VotingOptionForVotingClassifierError
        except VotingOptionForVotingClassifierError as e:
            print(e.get_message())
            print("Error code: " + e.get_code())
        else:
            try:
                if not self.classifier_option_check(estimator_1, self.classifier.ESTIMATORS_AND_CLASSIFIERS) \
                        or not self.classifier_option_check(estimator_2, self.classifier.ESTIMATORS_AND_CLASSIFIERS)\
                        or not self.classifier_option_check(estimator_3, self.classifier.ESTIMATORS_AND_CLASSIFIERS):
                    raise EstimatorOptionError
            except EstimatorOptionError as e:
                print(e.get_message())
                print("Error code: " + e.get_code())
            else:
                self.classifier.set_vc_clf_1(self.classifier.ESTIMATORS_AND_CLASSIFIERS[estimator_1]())
                self.classifier.set_vc_clf_2(self.classifier.ESTIMATORS_AND_CLASSIFIERS[estimator_2]())
                self.classifier.set_vc_clf_3(self.classifier.ESTIMATORS_AND_CLASSIFIERS[estimator_3]())
                self.classifier.set_clf_vtc()