from dataFile import *
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier,BaggingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
ext = ExtraTreesClassifier(
    n_estimators=1000,max_features='sqrt',max_depth=10,bootstrap=True
)
dec = DecisionTreeClassifier(max_depth=10,splitter="best",max_features="sqrt")
ens = BaggingClassifier(base_estimator=dec,n_estimators=1000,bootstrap=True)

from sklearn.metrics import accuracy_score
ext = ext.fit(X,np.ravel(y))
ext_pred = ext.predict(X_test)
ens = ens.fit(X,np.ravel(y))
ens_pred = ens.predict(X_test)

print(accuracy_score(y_test,ens_pred))
print(accuracy_score(y_test,ext_pred))