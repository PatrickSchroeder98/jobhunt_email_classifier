from src.interface.emailclassifierapp import EmailClassifierApp

"""This example shows the usage of 3-stage-classifier. User can view the accuracy of models and predictions."""

app = EmailClassifierApp()

emails = [
    """(...)

Thanks for your application for the Example Job position at Example Company. 

We’re excited to invite you to the first stage of the interview process!

During this interview, we'll get to know each other a little better and 
you'll have the opportunity to learn more about our company and the role.

We have a few dates and times available for the interview, 
so let us know which one works best for you: 

(...)
 """,
    """(...)

Click the link below and claim your personal discount on all games in our store.

The autumn sale starts today. This link will be valid until November 30th.

(...)
""",
]

app.train_3_stage_pipelines()

print("Accuracy of pipelines" + "\n")
app.view_3_stage_pipelines_accuracy()

print(emails[0])
print(app.classify_emails_3_stage_pipelines(emails)[0])


print(emails[1])
print(app.classify_emails_3_stage_pipelines(emails)[1])
