from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        X["Temp_diff"] = X["Process temperature [K]"] - X["Air temperature [K]"]
        X["Power_load"] = X["Torque [Nm]"] * X["Rotational speed [rpm]"]
        X["Wear_load_ratio"] = X["Tool wear [min]"] / (X["Torque [Nm]"] + 1)

        return X