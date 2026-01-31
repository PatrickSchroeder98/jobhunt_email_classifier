from src.interface.emailclassifierapp import EmailClassifierApp

"""This example shows the usage of multiclassifier. User can view the accuracy of models and predictions.
It also shows the usage of non-default classifier option - voting classifier with estimators."""

app = EmailClassifierApp()

emails = [
    """(...)

Thanks for your application for the Example Job position at Example Company. 

Unfortunately after careful consideration we will not be moving forward with your application.

We wish you the best of luck in your job search and would encourage you to apply for future
 
positions that may be a better match for your skills and experience. 

(...)
 """,
    """(...)


We hope this email finds you well.

We have noticed that you didn't log in to your user account since February.

In the meanwhile our data policy has changed, you can read the new terms in the link below: 

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
