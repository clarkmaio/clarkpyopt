import cvxpy as cvx

from dataclasses import dataclass
import pandas as pd
import numpy as np
from typing import List, Union, Iterable
import pickle

@dataclass
class Result:
    weights: np.ndarray
    intercet: float
    value: float

@dataclass
class Blend:
    convex: bool = True
    fit_intercept: bool = True
    lower_bound: Union[float, List[float]] = 0.
    upper_bound: Union[float, List[float]] = 1.
    lp_norm: int = 1

    def __post_init__(self):
        self.__intercept = None
        self.__coef = None
        self.__constraints = None
        self.__variable: cvx.Variable = None
        self.__is_trained: bool = False

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series], sample_weights: Iterable = None) -> Result:
        '''
        Fit the model
        '''

        # Check input
        assert X.shape[0] == y.shape[0], 'X and y must have the same number of rows'
        assert X.shape[1] > 0, 'X must have at least one column'

        if isinstance(X, pd.DataFrame):
            X = X.values

        if isinstance(y, pd.Series):
            y = y.values


        # Define variables
        n = X.shape[1] + self.fit_intercept
        self.__variable = cvx.Variable(n)

        # Build yhat
        y_hat = self._build_yhat(X)

        # Define objective
        if sample_weights is None:
            objective = cvx.Minimize(cvx.norm(y_hat - y, p=self.lp_norm))
        else:
            objective = cvx.Minimize(cvx.norm(cvx.multiply(sample_weights, y_hat - y), p=self.lp_norm))

        # Define constraints
        self.__constraints = self._build_constraints()

        # Solve problem
        problem = cvx.Problem(objective=objective, constraints=self.__constraints)
        problem.solve()

        # Return result
        if self.fit_intercept:
            self.__intercept = self.__variable.value[0]
            self.__coef = self.__variable.value[1:]
        else:
            self.__intercept = 0
            self.__coef = self.__variable.value

        self.__is_trained = True
        return Result(weights=self.__coef, intercet = self.__intercept, value=problem.value)

    def predict(self, X: np.ndarray) -> np.ndarray:
        '''
        Predict y
        '''

        if self.fit_intercept:
            return X @ self.__coef + self.__intercept
        else:
            return X @ self.__coef

    def _build_constraints(self):
        constraints = []

        if self.lower_bound is not None:
            if self.fit_intercept:
                constraints.append(self.__variable[1:] >= self.lower_bound)
            else:
                constraints.append(self.__variable >= self.lower_bound)

        if self.upper_bound is not None:
            if self.fit_intercept:
                constraints.append(self.__variable[1:] <= self.upper_bound)
            else:
                constraints.append(self.__variable <= self.upper_bound)

        if self.convex:
            if self.fit_intercept:
                constraints.append(cvx.sum(self.__variable[1:]) == 1)
            else:
                constraints.append(cvx.sum(self.__variable) == 1)

        return constraints

    def _build_yhat(self, X: np.ndarray) -> np.ndarray:
        '''
        Build yhat
        '''
        if self.fit_intercept:
            y_hat = X @ self.__variable[1:] + self.__variable[0]
        else:
            y_hat = X @ self.__variable

        return y_hat

    def save(self, path: str) -> None:
        '''
        Save model
        '''

        model_params = {
            '__intercept': self.__intercept,
            '__coef': self.__coef,
            '__constraints': self.__constraints,
            '__is_trained': self.__is_trained,

            'convex': self.convex,
            'fit_intercept': self.fit_intercept,
            'lower_bound': self.lower_bound,
            'upper_bound': self.upper_bound,
            'lp_norm': self.lp_norm

        }

        # Save pickle
        with open(path, 'wb') as f:
            pickle.dump(model_params, f)


    @staticmethod
    def load(path: str):
        '''
        Load model
        '''

        with open(path, 'rb') as f:
            model_params = pickle.load(f)

        # Create instance and initialize
        blend_mdl = Blend()
        for key, item in model_params.itemize():
            blend_mdl.__setattr__(name=key, value=item)
        return blend_mdl


    @property
    def coef_(self):
        return self.__coef

    @property
    def intercept_(self):
        return self.__intercept

    @property
    def constraints_(self):
        return self.__constraints


if __name__ == '__main__':

    # Generate data
    x = np.linspace(0, 10, 100)
    y = np.sin(x) + np.random.randn(100) * .1


    # Build model mockup
    arrays = [(y * .5 + np.random.randn(100) * .1).reshape(-1, 1)-1,
              (y * 1.7 + np.random.randn(100) * .1).reshape(-1, 1)-1,
              (np.random.randn(100) * .5).reshape(-1, 1)]
    X = np.concatenate(arrays, axis=1)
    X = pd.DataFrame(X, columns=['A', 'B', 'C'])

    # Fit model
    model = Blend(fit_intercept=True, convex=True, lp_norm=1)
    result = model.fit(X, y)
    y_hat = model.predict(X)

    # Print results
    print('Coeff solution', result.weights)
    print('Intercept', result.intercet)
    print('Optimal value', result.value)
