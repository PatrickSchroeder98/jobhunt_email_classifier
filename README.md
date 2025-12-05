# Jobhunt Emails Classifier

This project focuses on the development of a Natural Language Processing (NLP) based email classifier designed to support job seekers in managing recruitment-related communication. Using Scikit-learn pipelines, the system automatically categorizes incoming emails into key classes such as Invitation, Rejection, Confirmation, or Other. The solution demonstrates the use of machine learning, TF-IDF vectorization, and multiple model architectures. It highlights both the practical application of NLP techniques and the implementation of clean, modular Python code for a real-world use case.

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
* Black
* Unittest

## Tests  
The project includes a comprehensive set of tests to ensure that all functionalities are working correctly.  

## Documentation  
The documentation can be found on the [Software Documentation Website](https://patrickschroeder98.github.io/software_documentation/jobhunt_email_classifier_docs/index.html).  
Or in the websie [repository](https://github.com/PatrickSchroeder98/software_documentation/tree/main/jobhunt_email_classifier_docs).  