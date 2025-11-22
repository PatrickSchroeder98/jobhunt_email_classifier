from src.interface.emailclassifierapp import EmailClassifierApp

"""This example shows the usage of multiclassifier. User can view the accuracy of models and predictions.
It also shows the usage of non-default classifier option - voting classifier with estimators."""

app = EmailClassifierApp()

emails = [
    """(...)

Thanks for your application for the Example Job position at Example Company. 

We will contact you if your profile matches our requirements

(...)
 """,
    """(...)


Join our new course platform.

We give 20% discount for new users!

(...)
""",
]

app.train_multiclassifier_pipeline()
app.view_multiclassifier_accuracy()
result = app.predict_with_multiclassifier(emails)

print(emails[0])
print(result[0])


print(emails[1])
print(result[1])

app.train_multiclassifier_pipeline(classifier_option="VotingClassifier", estimator_1="KNeighborsClassifier", estimator_2="BernoulliNB", estimator_3="SVC")
app.view_multiclassifier_accuracy()

print(emails[0])
print(result[0])


print(emails[1])
print(result[1])
