# uploaderApp/Functions/regression.py
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import sklearn.metrics as metrics


def evaluate(model, data, selected_features):

    X = data[selected_features].values
    y = data.iloc[:, -1].values

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    test_mse_score = []
    test_mae_score = []
    test_rmse_score = []
    test_r2 = []

    train_mse_score = []
    train_mae_score = []
    train_rmse_score = []
    train_r2 = []

    for train_index, val_index in kf.split(X):
        x_train, x_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # Train-test split within the cross-validation loop
        x_train_split, x_test_split, y_train_split, y_test_split = train_test_split(x_train, y_train, test_size=0.2,
                                                                                    random_state=23)

        # Scaling data
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train_split)
        x_test_scaled = scaler.transform(x_test_split)

        model.fit(x_train_scaled, y_train_split)

        # Predictions
        y_train_pred = model.predict(x_train_scaled)
        y_test_pred = model.predict(x_test_scaled)

        test_mse_score.append(round(metrics.mean_squared_error(y_test_split, y_test_pred), 4))
        test_mae_score.append(round(metrics.mean_absolute_error(y_test_split, y_test_pred), 4))
        test_rmse_score.append(round(np.sqrt(metrics.mean_squared_error(y_test_split, y_test_pred)), 4))
        test_r2.append(round(metrics.r2_score(y_test_split, y_test_pred), 4))

        test_mse = np.mean(test_mse_score)
        test_mae = np.mean(test_mae_score)
        test_rmse = np.mean(test_rmse_score)
        test_r2_score = np.mean(test_r2)

        train_mse_score.append(round(metrics.mean_squared_error(y_train_split, y_train_pred), 4))
        train_mae_score.append(round(metrics.mean_absolute_error(y_train_split, y_train_pred), 4))
        train_rmse_score.append(round(np.sqrt(metrics.mean_squared_error(y_train_split, y_train_pred)), 4))
        train_r2.append(round(metrics.r2_score(y_train_split, y_train_pred), 4))

        train_mse = np.mean(train_mse_score)
        train_mae = np.mean(train_mae_score)
        train_rmse = np.mean(train_rmse_score)
        train_r2_score = np.mean(train_r2)

        result = {
            'Method': 'Unknown',
            'MAE': test_mae,
            'MSE': test_mse,
            'RMSE': test_rmse,
            'test_R_squared': test_r2_score,
        }
        result1 = {
            'Method': 'Unknown',
            'Train_MAE': train_mae,
            'Train_MSE': train_mse,
            'Train_RMSE': train_rmse,
            'Train_R_squared': train_r2_score
        }
        return result, result1


def linear_regression(data, selected_features):
    model = LinearRegression()
    result, result1 = evaluate(model, data, selected_features)
    result['Method'] = 'Multiple Linear Regression'
    result1['Method'] = 'Multiple Linear Regression'
    return result, result1


def random_forest_regression(data, selected_features):
    model = RandomForestRegressor(n_estimators=300, max_features='sqrt', max_depth=7, random_state=18)
    result, result1 = evaluate(model, data, selected_features)
    result['Method'] = 'Random Forest Regression'
    result1['Method'] = 'Random Forest Regression'
    return result, result1


def support_vector_regression(data, selected_features):
    model = SVR()
    result, result1 = evaluate(model, data, selected_features)
    result['Method'] = 'Support Vector Regression'
    result1['Method'] = 'Support Vector Regression'
    return result, result1


def ridge_regression(data, selected_features):
    model = Ridge(alpha=0.1)
    result, result1 = evaluate(model, data, selected_features)
    result['Method'] = 'Ridge Regression'
    result1['Method'] = 'Ridge Regression'
    return result, result1


def compare_methods(data, selected_features):
    linear_reg_result = linear_regression(data, selected_features)
    rf_result = random_forest_regression(data, selected_features)
    svr_result = support_vector_regression(data, selected_features)
    ridge_reg_result = ridge_regression(data, selected_features)

    # Create a dictionary to store the testing R-squared scores for each method
    test_r2_scores = {
        'Multiple Linear Regression': linear_reg_result[0]['test_R_squared'],
        'Random Forest Regression': rf_result[0]['test_R_squared'],
        'Support Vector Regression': svr_result[0]['test_R_squared'],
        'Ridge Regression': ridge_reg_result[0]['test_R_squared'],
    }

    # Find the method with the maximum testing R-squared score
    best_method = max(test_r2_scores, key=test_r2_scores.get)

    # Create separate DataFrames for training and testing performance metrics
    test_results_df = pd.DataFrame({
        'Method': [linear_reg_result[0]['Method'], rf_result[0]['Method'], svr_result[0]['Method'],
                   ridge_reg_result[0]['Method']],
        'MAE_test': [linear_reg_result[0]['MAE'], rf_result[0]['MAE'], svr_result[0]['MAE'],
                     ridge_reg_result[0]['MAE']],
        'MSE_test': [linear_reg_result[0]['MSE'], rf_result[0]['MSE'], svr_result[0]['MSE'],
                     ridge_reg_result[0]['MSE']],
        'RMSE_test': [linear_reg_result[0]['RMSE'], rf_result[0]['RMSE'], svr_result[0]['RMSE'],
                      ridge_reg_result[0]['RMSE']],
        'test_R_squared': [linear_reg_result[0]['test_R_squared'], rf_result[0]['test_R_squared'],
                           svr_result[0]['test_R_squared'], ridge_reg_result[0]['test_R_squared']],
    })

    train_results_df = pd.DataFrame({
        'Method': [linear_reg_result[1]['Method'], rf_result[1]['Method'], svr_result[1]['Method'],
                   ridge_reg_result[1]['Method']],
        'MAE_train': [linear_reg_result[1]['Train_MAE'], rf_result[1]['Train_MAE'], svr_result[1]['Train_MAE'],
                      ridge_reg_result[1]['Train_MAE']],
        'MSE_train': [linear_reg_result[1]['Train_MSE'], rf_result[1]['Train_MSE'], svr_result[1]['Train_MSE'],
                      ridge_reg_result[1]['Train_MSE']],
        'RMSE_train': [linear_reg_result[1]['Train_RMSE'], rf_result[1]['Train_RMSE'], svr_result[1]['Train_RMSE'],
                       ridge_reg_result[1]['Train_RMSE']],
        'Train_R_squared': [linear_reg_result[1]['Train_R_squared'], rf_result[1]['Train_R_squared'],
                            svr_result[1]['Train_R_squared'], ridge_reg_result[1]['Train_R_squared']],
    })

    return test_results_df, train_results_df, best_method
