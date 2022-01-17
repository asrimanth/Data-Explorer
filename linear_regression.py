'''
A module for implementing some of the techniques in Linear Regression.
Usage:
Call the object of the class LinearReg using the following parameters:

    LinearReg(fit_intercept=True, normalize=False, copy_X=True, n_jobs=None)
'''

import numpy as np
import pandas as pd

from sklearn import linear_model, metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

import statsmodels.api as sm


class LinearReg():

    '''A class which implements the 4 Regression techiniques.
        1) Ordinary Least Squares(OLS) from sklearn
        2) Ridge Regression
        3) LASSO Regression
        4) Ordinary Least Squares(OLS) from statsmodels

        Input for dataset : set_dataset(dataset,feature_list,target)
        Output for dataset : get_dataset()'''

    def __init__(self, fit_intercept=True, normalize=False, copy_X=True, n_jobs=None):
        '''The initialization of params used in sklearn Linear Regression.'''
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.n_jobs = n_jobs

    def set_dataset(self, dataset, feature_list, target):
        '''The method used to take the input dataset as a pandas DataFrame object.'''
        self.dataset = dataset
        features = self.dataset[feature_list]
        target = self.dataset[target]

        # Splitting the training and test sets.
        x_train, x_test, y_train, y_test = train_test_split(
            features, target, test_size=0.4, random_state=1)
        # For further use
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def get_dataset(self):
        return dataset

    def OLS_sklearn(self):
        '''Application of OLS from sklearn'''
        lin_reg = linear_model.LinearRegression(
            self.fit_intercept, self.normalize, self.copy_X, self.n_jobs)
        lin_reg.fit(self.x_train, self.y_train)
        mse_list = cross_val_score(lin_reg, self.x_train, self.y_train,
                                   scoring='neg_mean_squared_error', cv=5)
        mean_MSE = np.mean(mse_list)
        error_metrics = {'MSE': mse_list}
        return str(error_metrics)

    def ridge_regression(self):
        '''Application of Ridge Regression from sklearn'''
        alpha = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20]
        ridge = linear_model.Ridge(fit_intercept=self.fit_intercept,
                                   normalize=self.normalize, copy_X=self.copy_X)
        parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20]}
        ridge_regressor = GridSearchCV(ridge, parameters, scoring='neg_mean_squared_error', cv=5)
        ridge_regressor.fit(self.x_train, self.y_train)
        y_pred_ridge = ridge_regressor.predict(self.x_test)
        return 'Ridge best params : {}'.format(ridge_regressor.best_params_)
        print('Ridge best score : {}'.format(ridge_regressor.best_score_))

    def lasso_regression(self):
        '''Application of LASSO Regression from sklearn'''
        lasso = linear_model.Lasso(fit_intercept=self.fit_intercept,
                                   normalize=self.normalize, copy_X=self.copy_X, tol=0.01)
        parameters = {'alpha': [1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20, 25, 30, 35, 40]}
        lasso_regressor = GridSearchCV(lasso, parameters, scoring='neg_mean_squared_error', cv=5)
        lasso_regressor.fit(self.x_test, self.y_test)
        y_pred_lasso = lasso_regressor.predict(self.x_test)
        return 'Lasso best params : {}'.format(lasso_regressor.best_params_)
        print('Lasso best score : {}'.format(lasso_regressor.best_score_))

    def OLS_statsmodels(self):
        '''Application of Ordinary Least Squares(OLS) from statsmodels'''
        model1 = sm.OLS(self.y_train, self.x_train).fit()
        predictions1 = model1.predict(self.x_test)  # make the predictions by the model
        return model1.summary()
