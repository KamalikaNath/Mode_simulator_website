import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler


def stepwise_selection(data):
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    X = sm.add_constant(X)

    def adjusted_r2(r2, n, p):
        return 1 - (1 - r2) * ((n - 1) / (n - p - 1))

    included, excluded, best_adj_r2 = [], list(X.columns), -float('inf')

    while len(excluded) > 0:
        candidate_adj_r2 = [(feature, adjusted_r2(model.rsquared, X.shape[0], len(included) + 1))
                            for feature in excluded
                            if (model := sm.OLS(y, X[included + [feature]]).fit()).rsquared]

        best_candidate = max(candidate_adj_r2, key=lambda x: x[1])
        if best_candidate[1] > best_adj_r2:
            best_adj_r2 = best_candidate[1]
            included.append(best_candidate[0])
            excluded.remove(best_candidate[0])
        else:
            break

    final_model = sm.OLS(y, X[included]).fit()
    selected_features = final_model.model.exog_names[1:]
    summary_table = pd.DataFrame({'Features': selected_features,
                                  'P_Values': final_model.pvalues[1:].round(3)}).reset_index(drop=True)
    selected_features_df = summary_table[summary_table['P_Values'] < 0.05]
    selected_features_list = selected_features_df['Features'].tolist()
    # request.session['selected_features'] = selected_features_list

    return selected_features_df


def lasso_selection(data):
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    X_scaled = StandardScaler().fit_transform(X)

    lasso = Lasso(alpha=0.1)
    lasso.fit(X_scaled, y)

    selected_features_df = pd.DataFrame({'Features': data.columns[:-1], 'Coefficient': lasso.coef_.round(2)})
    selected_features_df['Importance'] = selected_features_df['Coefficient'].abs()
    selected_features_df = selected_features_df[selected_features_df['Coefficient'] != 0]
    selected_features_df.drop('Coefficient', axis=1, inplace=True)
    selected_features_df.reset_index(drop=True, inplace=True)
    selected_features_list = selected_features_df['Features'].tolist()
    # request.session['selected_features'] = selected_features_list

    return selected_features_df


def RF_selection(data):
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    feature_names = data.columns[:-1]

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)

    importances = rf.feature_importances_.round(2)
    indices = np.argsort(importances)[::-1]

    selected_features_df = pd.DataFrame({'Features': feature_names[indices], 'Importance': importances[indices]})
    selected_features_df = selected_features_df[selected_features_df['Importance'] != 0]
    selected_features_df.reset_index(drop=True, inplace=True)
    selected_features_list = selected_features_df['Features'].tolist()
    # request.session['selected_features'] = selected_features_list

    return selected_features_df


def ridge_selection(data):
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    feature_names = data.columns[:-1]

    X_train = StandardScaler().fit_transform(X)

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y)

    importances = np.abs(ridge.coef_.round(2))
    indices = np.argsort(importances)[::-1]

    importance_df = pd.DataFrame({'Features': feature_names, 'Importance': importances[indices]})
    selected_features_df = importance_df[importance_df['Importance'] > 0.01]
    selected_features_df.reset_index(drop=True, inplace=True)
    selected_features_list = selected_features_df['Features'].tolist()
    # request.session['selected_features'] = selected_features_list

    return selected_features_df


def feature_selection(df, feature_selection_method):
    if feature_selection_method == 'Stepwise Regression':
        selected_features_df = stepwise_selection(df)

    elif feature_selection_method == 'LASSO':
        selected_features_df = lasso_selection(df)

    elif feature_selection_method == 'Random Forest Regression':
        selected_features_df = RF_selection(df)

    elif feature_selection_method == 'Ridge Regression':
        selected_features_df = ridge_selection(df)

    else:
        raise ValueError("Invalid feature selection method")

    selected_features = selected_features_df.to_numpy().tolist()

    return selected_features
