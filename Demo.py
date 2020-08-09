from xgboost import XGBClassifier
import numpy as np
from sklearn.metrics import accuracy_score
from dataFile import *
model =XGBClassifier()
model.fit(X,np.ravel(y))
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
print("Accuracy Score: %.2f%%" %(accuracy*100.0))
