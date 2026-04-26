from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def train_lr(X, y):
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model

def train_rf(X, y):
    model = RandomForestClassifier()
    model.fit(X, y)
    return model