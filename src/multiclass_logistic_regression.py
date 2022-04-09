from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
from skfda import FDataGrid
from skfda.representation._typing import NDArrayInt, NDArrayAny
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression as mvLogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import check_classification_targets


class MulticlassLogisticRegression(BaseEstimator, ClassifierMixin):
    
    def __init__(self, p: int = 5) -> None:
        self.p = p

    def fit(self, X: FDataGrid, y: NDArrayAny) -> MulticlassLogisticRegression:
        X, classes, y_ind = self._argcheck_X_y(X, y)
        self.classes_ = classes
        
        n_samples = len(y)
        n_features = len(X.grid_points[0])
        
        selected_indexes = np.zeros(self.p, dtype=np.intc)
        
        mvlr = mvLogisticRegression(penalty='l2')
        
        x_mv = np.zeros((n_samples, self.p))
        LL = np.zeros(n_features)
        for q in range(self.p):
            for t in range(n_features):
                x_mv[:, q] = X.data_matrix[:, t, 0]
                mvlr.fit(x_mv[:, :q + 1], y_ind)
                
                # log-likelihood function at t
                log_probs = mvlr.predict_log_proba(x_mv[:, :q + 1])
                log_probs = np.concatenate(
                    (log_probs[y_ind == 0, 0], log_probs[y_ind == 1, 1]),
                )
                LL[t] = np.mean(log_probs)
            
            tmax = np.argmax(LL)
            selected_indexes[q] = tmax
            x_mv[:, q] = X.data_matrix[:, tmax, 0]
        
        # fit for the complete set of points
        mvlr.fit(x_mv, y_ind)

        self.coef_ = mvlr.coef_
        self.intercept_ = mvlr.intercept_
        self._mvlr = mvlr
        
        self._selected_indexes = selected_indexes
        self.points_ = X.grid_points[0][selected_indexes]
        
        return self
        
    def predict(self, X: FDataGrid) -> NDArrayInt:
        check_is_fitted(self)
        return self._wrapper(self._mvlr.predict, X)

    def predict_log_proba(self, X: FDataGrid) -> NDArrayInt:
        check_is_fitted(self)
        return self._wrapper(self._mvlr.predict_log_proba, X)

    def predict_proba(self, X: FDataGrid) -> NDArrayInt:
        check_is_fitted(self)
        return self._wrapper(self._mvlr.predict_proba, X)

    def _classifier_get_classes(self, y: np.ndarray) -> Tuple[np.ndarray, NDArrayInt]:
        check_classification_targets(y)

        le = LabelEncoder()
        y_ind = le.fit_transform(y)

        classes = le.classes_

        if classes.size < 2:
            raise ValueError(
                f'The number of classes has to be greater than'
                f'one; got {classes.size} class',
            )
        return classes, y_ind

    def _argcheck_X(self, X: FDataGrid) -> FDataGrid:
        if X.dim_domain > 1:
            raise ValueError(
                f'The dimension of the domain has to be one'
                f'; got {X.dim_domain} dimensions',
            )

        return X

    def _argcheck_X_y(self, X: FDataGrid, y: NDArrayAny) -> Tuple[FDataGrid, NDArrayAny, NDArrayAny]:
        X = self._argcheck_X(X)
        classes, y_ind = self._classifier_get_classes(y)

        if classes.size <= 2:
            raise ValueError(
                f'The number of classes has to be more than two'
                f'; got {classes.size} classes',
            )

        if (len(y) != len(X)):
            raise ValueError(
                "The number of samples on independent variables"
                " and classes should be the same",
            )

        return (X, classes, y_ind)

    def _wrapper(self, method: Callable[[NDArrayAny], NDArrayAny], X: FDataGrid) -> NDArrayAny:
        """Wrap multivariate logistic regression method.

        This function transforms functional data in order to pass
        them to a multivariate logistic regression method.

        .. warning::
            This function can't be called before fit.
        """
        X = self._argcheck_X(X)
        x_mv = X.data_matrix[:, self._selected_indexes, 0]
        return method(x_mv)
