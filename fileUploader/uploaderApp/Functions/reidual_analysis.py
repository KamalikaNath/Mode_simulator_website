import base64

import matplotlib.pyplot as plt
from scipy.stats import shapiro, probplot
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from statsmodels.stats.stattools import durbin_watson


def get_base64_encoded_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')


def MLR_residual_diagnostics(data, selected_features):
    X = data[selected_features].values
    y = data.iloc[:, -1].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)

    reg = LinearRegression()
    reg.fit(x_train, y_train)

    # Make predictions on the test set
    y_test_pred = reg.predict(x_test)
    residuals = y_test - y_test_pred

    # Create a 2x2 grid for subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    durbin_watson_statistic = durbin_watson(residuals)
    if durbin_watson_statistic < 1.5:
        durbin_watson_result = "Positive autocorrelation is present among residuals."
    elif 1.5 < durbin_watson_statistic < 2.5:
        durbin_watson_result = "No autocorrelation is present among residuals."
    else:
        durbin_watson_result = "Negative autocorrelation is present among residuals."

    axs[0, 0].text(0.1, 0.5,
                   f"Durbin-Watson Test:\nNull Hypothesis: Residuals are not autocorrelated.\nTest Statistic(d) "
                   f"(range (0,4)): {durbin_watson_statistic:.4f}\nTest Result: {durbin_watson_result}", fontsize=16)
    axs[0, 0].axis("off")

    # Residuals Histogram
    axs[0, 1].hist(residuals, bins=20, edgecolor='black')
    axs[0, 1].set_xlabel("Residuals")
    axs[0, 1].set_ylabel("Frequency")
    axs[0, 1].set_title(f"Residuals Histogram")

    # Shapiro-Wilk test
    _, p_value = shapiro(residuals)
    if p_value < 0.05:
        shapiro_result = "Null hypothesis is rejected.\nResiduals are significantly different from normal " \
                         "distribution. "
    else:
        shapiro_result = "Null hypothesis is accepted.\nResiduals are Normally Distributed."
    axs[1, 0].text(0.1, 0.5, f"Shapiro-Wilk Test:\nNull Hypothesis: Residuals are normally distributed (P-value>0.05)."
                             f"\np-value: {p_value:.4f}\n{shapiro_result}", fontsize=16)
    axs[1, 0].axis("off")

    # QQ plot
    probplot(residuals, plot=axs[1, 1])
    axs[1, 1].set_title(f"QQ Plot")

    plt.tight_layout()
    img_path = "static/images/MLR_residual_plot.png"
    plt.savefig(img_path)
    plt.close()

    return img_path


def SVR_residual_diagnostics(data, selected_features):
    X = data[selected_features].values
    y = data.iloc[:, -1].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)

    # y_predicted = linear_regression(data)['y_pred']
    reg = SVR()
    reg.fit(x_train, y_train)

    # Make predictions on the test set
    y_test_pred = reg.predict(x_test)
    residuals = y_test - y_test_pred

    # Create a 2x2 grid for subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    durbin_watson_statistic = durbin_watson(residuals)
    if durbin_watson_statistic < 1.5:
        durbin_watson_result = "Positive autocorrelation is present among residuals."
    elif 1.5 < durbin_watson_statistic < 2.5:
        durbin_watson_result = "No autocorrelation is present among residuals."
    else:
        durbin_watson_result = "Negative autocorrelation is present among residuals."

    axs[0, 0].text(0.1, 0.5,
                   f"Durbin-Watson Test:\nNull Hypothesis: Residuals are not autocorrelated.\nTest Statistic(d) "
                   f"(range (0,4)): {durbin_watson_statistic:.4f}\nTest Result: {durbin_watson_result}", fontsize=16)
    axs[0, 0].axis("off")

    # Residuals Histogram
    axs[0, 1].hist(residuals, bins=20, edgecolor='black')
    axs[0, 1].set_xlabel("Residuals")
    axs[0, 1].set_ylabel("Frequency")
    axs[0, 1].set_title(f"Residuals Histogram")

    # Shapiro-Wilk test
    _, p_value = shapiro(residuals)
    if p_value < 0.05:
        shapiro_result = "Null hypothesis is rejected.\nResiduals are significantly different from normal " \
                         "distribution. "
    else:
        shapiro_result = "Null hypothesis is accepted.\nResiduals are Normally Distributed."
    axs[1, 0].text(0.1, 0.5, f"Shapiro-Wilk Test:\nNull Hypothesis: Residuals are normally distributed (P-value>0.05)."
                             f"\np-value: {p_value:.4f}\n{shapiro_result}", fontsize=16)
    axs[1, 0].axis("off")

    # QQ plot
    probplot(residuals, plot=axs[1, 1])
    axs[1, 1].set_title(f"QQ Plot")

    plt.tight_layout()
    img_path = "static/images/SVR_residual_plot.png"
    plt.savefig(img_path)
    plt.close()

    # return base64_image
    return img_path


def RF_residual_diagnostics(data, selected_features):
    X = data[selected_features].values
    y = data.iloc[:, -1].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)

    # y_predicted = linear_regression(data)['y_pred']
    reg = RandomForestRegressor(n_estimators=300, max_features='sqrt', max_depth=7, random_state=18)
    reg.fit(x_train, y_train)

    # Make predictions on the test set
    y_test_pred = reg.predict(x_test)
    residuals = y_test - y_test_pred

    # Create a 2x2 grid for subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    durbin_watson_statistic = durbin_watson(residuals)
    if durbin_watson_statistic < 1.5:
        durbin_watson_result = "Positive autocorrelation is present among residuals."
    elif 1.5 < durbin_watson_statistic < 2.5:
        durbin_watson_result = "No autocorrelation is present among residuals."
    else:
        durbin_watson_result = "Negative autocorrelation is present among residuals."

    axs[0, 0].text(0.1, 0.5,
                   f"Durbin-Watson Test:\nNull Hypothesis: Residuals are not autocorrelated.\nTest Statistic(d) "
                   f"(range (0,4)): {durbin_watson_statistic:.4f}\nTest Result: {durbin_watson_result}", fontsize=16)
    axs[0, 0].axis("off")

    # Residuals Histogram
    axs[0, 1].hist(residuals, bins=20, edgecolor='black')
    axs[0, 1].set_xlabel("Residuals")
    axs[0, 1].set_ylabel("Frequency")
    axs[0, 1].set_title(f"Residuals Histogram")

    # Shapiro-Wilk test
    _, p_value = shapiro(residuals)
    if p_value < 0.05:
        shapiro_result = "Null hypothesis is rejected.\nResiduals are significantly different from normal " \
                         "distribution. "
    else:
        shapiro_result = "Null hypothesis is accepted.\nResiduals are Normally Distributed."
    axs[1, 0].text(0.1, 0.5, f"Shapiro-Wilk Test:\nNull Hypothesis: Residuals are normally distributed (P-value>0.05)."
                             f"\np-value: {p_value:.4f}\nTest Result:{shapiro_result}", fontsize=16)
    axs[1, 0].axis("off")

    # QQ plot
    probplot(residuals, plot=axs[1, 1])
    axs[1, 1].set_title(f"QQ Plot")

    plt.tight_layout()
    img_path = "static/images/RF_residual_plot.png"
    plt.savefig(img_path)
    plt.close()

    return img_path


def Ridge_residual_diagnostics(data, selected_features):
    X = data[selected_features].values
    y = data.iloc[:, -1].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)

    # y_predicted = linear_regression(data)['y_pred']
    reg = Ridge(alpha=1)
    reg.fit(x_train, y_train)

    # Make predictions on the test set
    y_test_pred = reg.predict(x_test)
    residuals = y_test - y_test_pred

    # Create a 2x2 grid for subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    durbin_watson_statistic = durbin_watson(residuals)
    if durbin_watson_statistic < 1.5:
        durbin_watson_result = "Positive autocorrelation is present among residuals."
    elif 1.5 < durbin_watson_statistic < 2.5:
        durbin_watson_result = "No autocorrelation is present among residuals."
    else:
        durbin_watson_result = "Negative autocorrelation is present among residuals."

    axs[0, 0].text(0.1, 0.5,
                   f"Durbin-Watson Test:\nNull Hypothesis: Residuals are not autocorrelated.\nTest Statistic(d) "
                   f"(range (0,4)): {durbin_watson_statistic:.4f}\nTest Result: {durbin_watson_result}", fontsize=16)
    axs[0, 0].axis("off")

    # Residuals Histogram
    axs[0, 1].hist(residuals, bins=20, edgecolor='black')
    axs[0, 1].set_xlabel("Residuals")
    axs[0, 1].set_ylabel("Frequency")
    axs[0, 1].set_title(f"Residuals Histogram")

    # Shapiro-Wilk test
    _, p_value = shapiro(residuals)
    if p_value < 0.05:
        shapiro_result = "Null hypothesis is rejected.\nResiduals are significantly different from normal " \
                         "distribution. "
    else:
        shapiro_result = "Null hypothesis is accepted.\nResiduals are Normally Distributed."
    axs[1, 0].text(0.1, 0.5, f"Shapiro-Wilk Test:\nNull Hypothesis: Residuals are normally distributed (P-value>0.05)."
                             f"\np-value: {p_value:.4f}\nTest Result:{shapiro_result}", fontsize=16)
    axs[1, 0].axis("off")

    # QQ plot
    probplot(residuals, plot=axs[1, 1])
    axs[1, 1].set_title(f"QQ Plot")

    plt.tight_layout()
    img_path = "static/images/Ridge_residual_plot.png"
    plt.savefig(img_path)
    plt.close()

    return img_path
