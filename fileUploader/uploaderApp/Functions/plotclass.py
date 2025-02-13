import matplotlib

import matplotlib.pyplot as plt
from scipy.stats import probplot, shapiro
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

matplotlib.use('Agg')


def plot_linear_regression(data, selected_features):
    X = data[selected_features].values
    y = data.iloc[:, -1].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)

    # y_predicted = linear_regression(data)['y_pred']
    linear_reg = LinearRegression()
    linear_reg.fit(x_train, y_train)

    # Make predictions on the test set
    y_predicted = linear_reg.predict(x_test)

    matplotlib.use('Agg')
    fig, ax = plt.subplots()

    # Plotting
    x = list(range(len(y_test)))
    ax.plot(x, y_test,  linestyle='-', color='red', label='Actual', alpha=0.5)
    ax.plot(x, y_predicted, linestyle='-', color='blue', label='Predicted', alpha=0.5)
    ax.set_xlabel('Data points')
    ax.set_ylabel('Yield (t ha-1)')
    ax.set_title('Multiple Linear Regression: Actual vs Predicted')
    ax.legend()
    ax.grid(True)

    # Return the plot object
    return plt


def plot_random_forest(data, selected_features):
    X = data[selected_features].values
    y = data.iloc[:, -1].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)
    rf = RandomForestRegressor(n_estimators=300, max_features='sqrt', max_depth=7, random_state=18)
    rf.fit(x_train, y_train)
    # Predict on test data
    y_predicted = rf.predict(x_test)

    matplotlib.use('Agg')
    fig, ax = plt.subplots()

    # Plotting
    x = list(range(len(y_test)))
    ax.plot(x, y_test,  linestyle='-', color='red', label='Actual', alpha=0.5)
    ax.plot(x, y_predicted,  linestyle='-', color='blue', label='Predicted', alpha=0.5)
    ax.set_xlabel('Data points')
    ax.set_ylabel('Yield (t ha-1)')
    ax.set_title('Random Forest Regression: Actual vs Predicted')
    ax.legend()
    ax.grid(True)

    # Return the plot object
    return plt


def plot_support_vector(data, selected_features):
    X = data[selected_features].values
    y = data.iloc[:, -1].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)
    # y_predicted = support_vector_regression(data)['y_pred']
    model = SVR()
    model.fit(x_train, y_train)
    # Predict on test data
    y_predicted = model.predict(x_test)

    matplotlib.use('Agg')
    fig, ax = plt.subplots()

    # Plotting
    x = list(range(len(y_test)))
    ax.plot(x, y_test,  linestyle='-', color='red', label='Actual', alpha=0.5)
    ax.plot(x, y_predicted, linestyle='-', color='blue', label='Predicted', alpha=0.5)
    ax.set_xlabel('Data points')
    ax.set_ylabel('Yield (t ha-1)')
    ax.set_title('Support Vector Regression: Actual vs Predicted')
    ax.legend()
    ax.grid(True)

    # Return the plot object
    return plt


def plot_ridge(data, selected_features):
    X = data[selected_features].values
    y = data.iloc[:, -1].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)
    # Create a Ridge regression model with alpha=1 (you can adjust this hyperparameter)
    ridge = Ridge(alpha=1)

    # Fit the model on the training data
    ridge.fit(x_train, y_train)

    # Make predictions on the test data
    y_predicted = ridge.predict(x_test)

    matplotlib.use('Agg')
    fig, ax = plt.subplots()

    # Plotting
    x = list(range(len(y_test)))
    ax.plot(x, y_test, linestyle='-', color='red', label='Actual', alpha=0.5)
    ax.plot(x, y_predicted, linestyle='-', color='blue', label='Predicted', alpha=0.5)
    ax.set_xlabel('Data points')
    ax.set_ylabel('Yield (t ha-1)')
    ax.set_title('Ridge Regression: Actual vs Predicted')
    ax.legend()
    ax.grid(True)

    # Return the plot object
    return plt
