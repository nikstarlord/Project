import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.matlib import rand
from pandas import DataFrame
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve , auc
from sklearn.ensemble import ExtraTreesClassifier
from dataFile import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier,RandomForestClassifier
from sklearn.metrics import accuracy_score
stump = DecisionTreeClassifier(max_depth=10, splitter='best', max_features="sqrt")
ens = RandomForestClassifier()

#creating ensemble
ensmeble = BaggingClassifier(base_estimator=ens, n_estimators=1000,bootstrap=True)
ensmeble.fit(X,np.ravel(y))
y_pred_ensemble = ensmeble.predict(X_test)
print(accuracy_score(y_test,y_pred_ensemble))