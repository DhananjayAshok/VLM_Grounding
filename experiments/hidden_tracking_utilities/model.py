from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pickle

class Linear():
    def __init__(self, penalty='l2', C=1.0, scale=True):
        self.scale = scale
        if scale:
            self.scaler = StandardScaler()
        self.name = f"linear-{penalty}-C-{C}"
        self.model = LogisticRegression(random_state=0, penalty=penalty, C=C, class_weight="balanced")


    def fit(self, X_train, y_train):
        if self.scale:
            X_train = self.scaler.fit_transform(X_train)
        self.model.fit(X_train, y_train)

    def score(self, X_train, y_train):
        if self.scale:
            X_train = self.scaler.transform(X_train)
        return self.model.score(X_train, y_train)
    
    def predict_proba(self, X):
        if self.scale:
            X = self.scaler.transform(X)
        proba = self.model.predict_proba(X)
        return proba
    
    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)
    
    def save(self, path, name="model"):
        with open(path+f"/{name}.pkl", "wb") as f:
            pickle.dump(self.model, f)
    
    def load(self, path, name="model"):
        with open(path+f"/{name}.pkl", "rb") as f:
            self.model = pickle.load(f) 

    def __str__(self):
        return self.name
    
    def __repr__(self):
        return self.name