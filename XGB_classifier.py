import pandas as pd
import pickle

class pheno_classifier():
    
    def __init__(self):
        
        path = "XGB_classifier.pkl"
        with open(path, "rb") as model_file:
            clf = pickle.load(model_file)
            
        self.clf = clf
        
    def predict(self, X):
        prediction_prob = self.clf.predict_proba(X)
        return prediction_prob

X_test = pd.read_csv("X_test.csv")
print("Shape of df :{}".format(X_test.shape))

classifier = pheno_classifier()
y_prob = classifier.predict(X_test)

print("Prediction Probability are : {}".format(y_prob))