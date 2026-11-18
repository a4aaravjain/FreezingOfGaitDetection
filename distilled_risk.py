"""Probability interface for a fixed single-ankle distillation regressor."""
import numpy as np


class DistilledRisk:
    def __init__(self,regressor):
        self.regressor = regressor
    def predict_proba(self,x):
        p = np.clip(self.regressor.predict(x),0,1)
        return np.column_stack([1-p,p])
