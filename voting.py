from dataFile import *
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier,BaggingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
estimator = []
estimator.append(('ext',ExtraTreesClassifier(n_estimators=500,max_depth=10,max_features='sqrt',bootstrap=True)))
estimator.append(('dec',DecisionTreeClassifier(splitter="best",max_depth=10,max_features='sqrt')))
#estimator.append(('svc', SVC(kernel='rbf',probability=True)))
from sklearn.metrics import accuracy_score
vot_hard = VotingClassifier(estimators=estimator,voting='soft')
vot_hard.fit(X,np.ravel(y))
vot_pred = vot_hard.predict(X_test)
print(accuracy_score(y_test,vot_pred))
ens = BaggingClassifier(base_estimator=vot_hard,n_estimators=1000,bootstrap=True)
ens.fit(X,np.ravel(y))
ens_pred = ens.predict(X_test)

print(accuracy_score(y_test,ens_pred))