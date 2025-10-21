from src.classifiers.classifier import Classifier
from src.data_module.dataloader import DataLoader
from src.exceptions.exceptions import PathError, ClassifierOptionError, ModelNotFound
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
        self.CLASSIFIERS = {
            "MultinomialNB": self.classifier.set_clf_nb,
            "LogisticRegression": self.classifier.set_clf_lr,
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

    def classifier_option_check(self, option):
        """Method that checks if provided classifier option is valid."""
        return False if option not in self.CLASSIFIERS.keys() else True

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
                if not self.classifier_option_check(classifier_option):
                    raise ClassifierOptionError
                else:
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

