# Jobhunt Emails Classifier

This project presents a production-oriented Natural Language Processing (NLP) email classification system designed to assist job seekers in organizing and interpreting recruitment-related email communication. Built with Python and Scikit-learn, the application automatically classifies emails into meaningful categories such as Invitation, Rejection, Confirmation, or Non job-hunt related, significantly reducing manual effort during job searches. The system supports two complementary classification architectures: the three-stage binary pipelines, where emails are progressively filtered through specialized models (job-related detection → confirmation detection → invitation vs rejection); and a multiclassifier pipeline, capable of predicting all classes directly using a single model. Both approaches are implemented using modular Scikit-learn pipelines with TF-IDF vectorization and a flexible classifier registry supporting a wide range of algorithms, including Naive Bayes, Logistic Regression, SVMs, ensemble methods, and meta-classifiers such as VotingClassifier and StackingClassifier.

## Features  
* Emails Classification.
* Data loading functionality.
* Interface class.
* Functionality to choose NLP classifier.
* Functionality to choose solution architecture: 3-stage-classifier or multiclassifier

## Available classifiers
* MultinomialNB
* ComplementNB
* BernoulliNB
* LogisticRegression
* SGDClassifier
* RidgeClassifier
* LinearSVC
* SVC
* KNeighborsClassifier
* DecisionTreeClassifier
* ExtraTreeClassifier
* RandomForestClassifier
* GradientBoostingClassifier
* AdaBoostClassifier
* VotingClassifier (only for multiclassifier)
* StackingClassifier (only for multiclassifier)

## Technologies used
* Python programming language
* Scikit-learn NLP Modules
* Pandas
* Unittest
* Black
* Sphinx

## Tests  
The project includes a comprehensive set of tests to ensure that all functionalities are working correctly.  

## Documentation  
The documentation can be found on the [Software Documentation Website](https://patrickschroeder98.github.io/software_documentation/jobhunt_email_classifier_docs/index.html).  
Or in the websie [repository](https://github.com/PatrickSchroeder98/software_documentation/tree/main/jobhunt_email_classifier_docs).  