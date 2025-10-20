from src.classifiers.classifier import Classifier
from src.data_module.dataloader import DataLoader
from src.exceptions.exceptions import PathError
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
        self.CLASSIFIERS = ["MultinomialNB", "LogisticRegression"]
        self.CLASSIFIER_METHODS = [self.classifier.set_clf_nb, self.classifier.set_clf_lr]

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
        return False if option not in self.CLASSIFIERS else True