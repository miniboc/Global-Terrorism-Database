"""
    The following is a solution to correctly classify which organisation was responsible after an attack

    Problem: Multi-classification

    Algorithm: SVM

    Dataset: GTD (Global Terrorism Database).

    Author: Darren Smith

    ******NOTE******
    Must upgrade scikit-learn to run this program.
    Run command: conda update scikit-learn
    ******NOTE******
"""
# REMOVED USED IN INTIAL RESULTS ONLY
# ---------------------------------------------------------------
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn import tree
# from sklearn import naive_bayes
# ---------------------------------------------------------------

# REMOVED USED IN FEATURE SELECTION
# ---------------------------------------------------------------
# from sklearn.feature_selection import SelectPercentile
# from sklearn.feature_selection import f_regression
# from sklearn.ensemble import ExtraTreesClassifier
# ---------------------------------------------------------------

from sklearn import svm, model_selection
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
import itertools
import pandas as pd
import numpy as np
import time

# REMOVED USED IN INTIAL RESULTS ONLY
# ---------------------------------------------------------------
# Function will run each of the classifiers using cross fold validation
# def run_classifiers(data, target):
#
#     cross_svc = SVC()
#     scores = model_selection.cross_val_score(cross_svc, data, target, cv=10)
#     print "SVM : ", scores.mean()
#
#     dTree = tree.DecisionTreeClassifier()
#     scores = model_selection.cross_val_score(dTree, data, target, cv=10)
#     print "Tree : ", scores.mean()
#
#     nearestN = KNeighborsClassifier()
#     scores = model_selection.cross_val_score(nearestN, data, target, cv=10)
#     print "NNeighbour : ", scores.mean()
#
#     randomForest = RandomForestClassifier()
#     scores = model_selection.cross_val_score(randomForest, data, target, cv=10)
#     print "RForest : ", scores.mean()
#
#     nBayes = naive_bayes.GaussianNB()
#     scores = model_selection.cross_val_score(nBayes, data, target, cv=10)
#     print "Naive Bayes : ", scores.mean()
#
# ---------------------------------------------------------------


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("\n\nNormalized confusion matrix")
    else:
        print('\n\nConfusion matrix')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def main():
    # Runtime
    start_time = time.time()

    # read in dataset
    gtd = pd.read_csv("GTD/globalterrorismdb (cleaned).csv", delimiter=",")

    # remove features
    gtd = gtd.drop(['country_txt'], axis=1)
    gtd = gtd.drop(['region_txt'], axis=1)
    gtd = gtd.drop(['attacktype_txt'], axis=1)
    gtd = gtd.drop(['targtype_txt'], axis=1)
    gtd = gtd.drop(['targsubtype_txt'], axis=1)
    gtd = gtd.drop(['weaptype_txt'], axis=1)
    gtd = gtd.drop(['area'], axis=1)
    gtd = gtd.drop(['city'], axis=1)
    gtd = gtd.drop(['property'], axis=1)
    gtd = gtd.drop(['propextent'], axis=1)
    gtd = gtd.drop(['propextent_txt'], axis=1)

    # REMOVED USED TO DROP TO 2 FEATURES
    # ---------------------------------------------------------------
    # Dropped after feature selection
    # gtd = gtd.drop(['nwound'], axis=1)
    # gtd = gtd.drop(['ishostkid'], axis=1)
    # gtd = gtd.drop(['attacktype'], axis=1)
    # gtd = gtd.drop(['nkill'], axis=1)
    # gtd = gtd.drop(['targtype'], axis=1)
    # gtd = gtd.drop(['targsubtype'], axis=1)
    # gtd = gtd.drop(['weaptype'], axis=1)
    # gtd = gtd.drop(['year'], axis=1)
    # gtd = gtd.drop(['success'], axis=1)
    # ---------------------------------------------------------------

    # REMOVED USED IN INTIAL RESULTS ONLY
    # ---------------------------------------------------------------
    # gtd = gtd.drop(['country'], axis=1)
    # gtd = gtd.drop(['region'], axis=1)
    # gtd = gtd.drop(['attacktype'], axis=1)
    # gtd = gtd.drop(['targtype'], axis=1)
    # gtd = gtd.drop(['targsubtype'], axis=1)
    # gtd = gtd.drop(['weaptype'], axis=1)
    # ---------------------------------------------------------------

    # top organisations
    print "\n\nTop organisations"
    print pd.value_counts(gtd['gname'])

    # new dataframe with only selected organisations
    test1 = gtd[gtd.gname == 'Taliban']
    test2 = gtd[gtd.gname == 'Shining Path (SL)']
    test3 = gtd[gtd.gname == 'Farabundo Marti National Liberation Front (FMLN)']
    test4 = gtd[gtd.gname == 'Islamic State of Iraq and the Levant (ISIL)']
    test5 = gtd[gtd.gname == 'Irish Republican Army (IRA)']
    test6 = gtd[gtd.gname == 'Revolutionary Armed Forces of Colombia (FARC)']
    test7 = gtd[gtd.gname == 'New People\'s Army (NPA)']
    test8 = gtd[gtd.gname == 'Al-Shabaab']
    test9 = gtd[gtd.gname == 'Basque Fatherland and Freedom (ETA)']
    test10 = gtd[gtd.gname == 'Boko Haram']
    test11 = gtd[gtd.gname == 'Kurdistan Workers\' Party (PKK)']
    test12 = gtd[gtd.gname == 'Communist Party of India - Maoist (CPI-Maoist)']
    test13 = gtd[gtd.gname == 'Liberation Tigers of Tamil Eelam (LTTE)']
    test14 = gtd[gtd.gname == 'National Liberation Army of Colombia (ELN)']
    test15 = gtd[gtd.gname == 'Tehrik-i-Taliban Pakistan (TTP)']
    test16 = gtd[gtd.gname == 'Maoists']
    test17 = gtd[gtd.gname == 'Palestinians']
    test18 = gtd[gtd.gname == 'Nicaraguan Democratic Force (FDN)']
    test19 = gtd[gtd.gname == 'Al-Qaida in the Arabian Peninsula (AQAP)']
    test20 = gtd[gtd.gname == 'Manuel Rodriguez Patriotic Front (FPMR)']
    frames = [test1, test2, test3, test4, test5, test6, test7, test8, test9, test10, test11, test12, test13,
              test14, test15, test16, test17, test18, test19, test20]
    result = pd.concat(frames)

    # determine number of missing values in each column
    print "\n\nCheck missing values"
    print result.isnull().sum()

    # REMOVE TO RUN WITH 2 FEATURES
    # ---------------------------------------------------------------
    #impute the mean value for all missing values in the nkill, nwound, targsubtype, ishostkid column
    imputer = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=1)

    imputer.fit(result["nkill"])
    newValues1 = imputer.transform(result["nkill"])
    result["nkill"] = newValues1[0]

    imputer.fit(result["targsubtype"])
    newValues2 = imputer.transform(result["targsubtype"])
    result["targsubtype"] = newValues2[0]

    imputer.fit(result["nwound"])
    newValues3 = imputer.transform(result["nwound"])
    result["nwound"] = newValues3[0]

    imputer.fit(result["ishostkid"])
    newValues4 = imputer.transform(result["ishostkid"])
    result["ishostkid"] = newValues4[0]
    # ---------------------------------------------------------------

    print "\n\nCheck if missing values were removed"
    print result.isnull().sum()

    # encode categorical variables as continuous variables
    result['organisation'] = result['gname'].map({'Taliban': 0,
                                                  'Shining Path (SL)': 1,
                                                  'Farabundo Marti National Liberation Front (FMLN)': 2,
                                                  'Islamic State of Iraq and the Levant (ISIL)': 3,
                                                  'Irish Republican Army (IRA)': 4,
                                                  'Revolutionary Armed Forces of Colombia (FARC)': 5,
                                                  'New People\'s Army (NPA)': 6,
                                                  'Al-Shabaab': 7,
                                                  'Basque Fatherland and Freedom (ETA)': 8,
                                                  'Boko Haram': 9,
                                                  'Kurdistan Workers\' Party (PKK)': 10,
                                                  'Communist Party of India - Maoist (CPI-Maoist)': 11,
                                                  'Liberation Tigers of Tamil Eelam (LTTE)': 12,
                                                  'National Liberation Army of Colombia (ELN)': 13,
                                                  'Tehrik-i-Taliban Pakistan (TTP)': 14,
                                                  'Maoists': 15,
                                                  'Palestinians': 16,
                                                  'Nicaraguan Democratic Force (FDN)': 17,
                                                  'Al-Qaida in the Arabian Peninsula (AQAP)': 18,
                                                  'Manuel Rodriguez Patriotic Front (FPMR)': 19
                                                  }).astype(int)

    result = result.drop(['gname'], axis=1)

    print "\n\nData frame information"
    print result.info()

    # REMOVED USED IN INTIAL RESULTS ONLY
    # ---------------------------------------------------------------
    # perform one-hot encoding on Embarked column
    # result = pd.get_dummies(result, columns=["country_txt"])
    # result = pd.get_dummies(result, columns=["region_txt"])
    # result = pd.get_dummies(result, columns=["area"])
    # result = pd.get_dummies(result, columns=["city"])
    # result = pd.get_dummies(result, columns=["attacktype_txt"])
    # result = pd.get_dummies(result, columns=["targtype_txt"])
    # result = pd.get_dummies(result, columns=["targsubtype_txt"])
    # result = pd.get_dummies(result, columns=["weaptype_txt"])
    # --------------------------------------------------------------

    # Next separate the class data from the training data
    target = result["organisation"]
    data = result.drop(["organisation"], axis=1)

    # REMOVED FEATURE SELECTION
    # ---------------------------------------------------------------
    # Univariate Feature Selection
    # feature_names = list(result.columns.values)
    # Selector_f = SelectPercentile(f_regression, percentile=25)
    # Selector_f.fit(data, target)
    # for n, s in zip(feature_names, Selector_f.scores_):
    #     print 'F Score', s, "for feature", n

    # Tree-based Feature Selection
    # forest = ExtraTreesClassifier(n_estimators=250, random_state=0)
    # forest.fit(data, target)
    # importances = forest.feature_importances_
    # for n, s in zip(feature_names, importances):
    #     print 'F Score', s, "for feature", n
    # ---------------------------------------------------------------

    print "\n\nnumber of features"
    print len(result.columns)
    print "number of rows"
    print result.shape[0]

    # REMOVED USED IN INTIAL RESULTS ONLY
    # ---------------------------------------------------------------
    # print "\n\nRunning classifiers before standardization"
    # runClassifiers(data, target)
    # --------------------------------------------------------------

    # Run standardization on the data
    scalingObj = preprocessing.StandardScaler()
    standardizedData = scalingObj.fit_transform(data)
    data = pd.DataFrame(standardizedData, columns=data.columns)

    # Split the data into a training set and a test set
    X_train, X_test, y_train, y_test = train_test_split(data, target, random_state=0)

    # REMOVED USED AFTER RESULT ON TOP 10 (RESULT kernel = linear, C = 1)
    # Forcing best_params_ to above as I need the estimator object
    # ---------------------------------------------------------------
    # Hyper-parameter optimization on the data.
    print("\n\nRunning hyper-parameter optimization........")
    # param_grid = [{'kernel': ['rbf', 'poly', 'linear'], 'C': range(1, 15)}]
    param_grid = [{'kernel': ['linear'], 'C': range(1, 2)}]
    clf = GridSearchCV(SVC(), param_grid, cv=10)
    clf.fit(data, target)
    print("\n\nBest parameters set found on development set:")
    print(clf.best_params_)
    # ---------------------------------------------------------------

    # Run classifier
    classifier = svm.SVC(kernel=clf.best_params_["kernel"], C=clf.best_params_["C"])
    y_pred = classifier.fit(X_train, y_train).predict(X_test)

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)

    # Plot non-normalized confusion matrix
    plt.figure()
    class_names = ['Taliban', '(SL)', '(FMLN)', 'ISIL)', '(IRA)', '(FARC)',
                   '(NPA)', 'Al-Shabaab', '(ETA)', 'Boko Haram', '(PKK)',
                   '(CPI-Maoist)', '(LTTE)', '(ELN)', '(TTP)', 'Maoists',
                   'Palestinians', '(FDN)', '(AQAP', '(FPMR)']
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='Confusion matrix, without normalization')

    plt.show()

    # REMOVED USED IN INTIAL RESULTS ONLY
    # ---------------------------------------------------------------
    # print "\n\nRunning classifiers after standardization"
    # run_classifiers(data, target)
    # ---------------------------------------------------------------

    scores = model_selection.cross_val_score(clf.best_estimator_, data, target, cv=10)
    print "SVM : ", scores.mean()

    # Runtime
    print ("--- %s seconds ---" % (time.time() - start_time))


main()
