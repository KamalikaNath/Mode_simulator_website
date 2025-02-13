# Create your views here.
import os
import random
import string

import matplotlib.pyplot

from .Functions.regression import *
from .Functions.reidual_analysis import *
from django.http import HttpResponse
from .forms import UploadFileForm
from .Functions.plotclass import *
from .Functions.feature_class import *
import pandas as pd
import matplotlib.pyplot as plt
from urllib.parse import quote
import numpy as np
from django.shortcuts import render
from statsmodels.stats.stattools import durbin_watson
from scipy.stats import shapiro, probplot
import io
import base64


def home(request):
    return render(request, 'home.html')


def read_uploaded_file(file_path):
    df = pd.read_csv(file_path)  # Read the uploaded CSV file into a pandas DataFrame
    return df


def select_features(request):
    if request.method == 'POST':
        uploaded_file = request.FILES['file']
        feature_selection_method = request.POST.get('feature_selection_method')

        df = read_uploaded_file(uploaded_file)

        # Reset the index to a default integer index
        df.reset_index(drop=True, inplace=True)

        selected_features = feature_selection(df, feature_selection_method)

        csv_file_path = os.path.join('uploads', uploaded_file.name)
        df.to_csv(csv_file_path, index=False)
        # Store the CSV file name (without path) in the session
        request.session['csv_file_name'] = uploaded_file.name

        return render(request, 'uploaderApp/features_result.html', {
            'selected_features': selected_features,
            'num': len(selected_features),
            'method': feature_selection_method})
    else:
        form = UploadFileForm()

    return render(request, 'uploaderApp/features.html', {'form': form})


# Convert the plot image to base64 string for embedding in HTML
def save_plot(plot_func, df, selected_features):
    buffer = io.BytesIO()
    plot_func(df, selected_features)
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    return quote(image_base64)


def get_base64_encoded_image(image_path):
    with open(image_path, "rb") as img_file:
        encoded_image = base64.b64encode(img_file.read()).decode('utf-8')
    return f"data:image/png;base64,{encoded_image}"


def perform_regression(request):
    if request.method == 'POST':
        selected_features = request.POST.getlist('selected_features')
        regression_method = request.POST.get('regression_method')

        # Get the CSV file name from the session
        csv_file_name = request.session.get('csv_file_name')

        # Ensure the CSV file information is available in the session
        if csv_file_name is None:
            return HttpResponse("CSV file information not found in the session.")

        # Concatenate the path and filename for reading the file
        csv_file_path = f"uploads/{csv_file_name}"

        df = read_uploaded_file(csv_file_path)

        if regression_method == 'Multiple Linear Regression':
            result, result1 = linear_regression(df, selected_features)
            method = 'Multiple Linear Regression'
            plot_image = save_plot(plot_linear_regression, df, selected_features)

        elif regression_method == 'Random Forest Regression':
            result, result1 = random_forest_regression(df, selected_features)
            method = 'Random Forest Regression'
            plot_image = save_plot(plot_random_forest, df, selected_features)

        elif regression_method == 'Support Vector Regression':
            result, result1 = support_vector_regression(df, selected_features)
            method = 'Support Vector Regression'
            plot_image = save_plot(plot_support_vector, df, selected_features)

        elif regression_method == 'Ridge Regression':
            result, result1 = ridge_regression(df, selected_features)
            method = 'Ridge Regression'
            plot_image = save_plot(plot_ridge, df, selected_features)

        else:
            test_results_df, train_results_df, best_method = compare_methods(df, selected_features)

            # Generate actual vs predicted plot for the best method
            if best_method == 'Multiple Linear Regression':
                plot_image = save_plot(plot_linear_regression, df, selected_features)
                residual_plot = MLR_residual_diagnostics(df, selected_features)
                print("Residual Plot:", residual_plot)  # Print the residual plot for debugging

            elif best_method == 'Random Forest Regression':
                plot_image = save_plot(plot_random_forest, df, selected_features)
                residual_plot = RF_residual_diagnostics(df, selected_features)
                print("Residual Plot:", residual_plot)  # Print the residual plot for debugging

            elif best_method == 'Support Vector Regression':
                plot_image = save_plot(plot_support_vector, df, selected_features)
                residual_plot = SVR_residual_diagnostics(df, selected_features)
                print("Residual Plot:", residual_plot)  # Print the residual plot for debugging

            elif best_method == 'Ridge Regression':
                plot_image = save_plot(plot_ridge, df, selected_features)
                residual_plot = Ridge_residual_diagnostics(df, selected_features)
                print("Residual Plot:", residual_plot)  # Print the residual plot for debugging

            else:
                plot_image = None
                residual_plot = None

                # Pass the results, best method, and plot image to the template context
            context = {
                'test_results_df': test_results_df,
                'train_results_df': train_results_df,
                'best_method': best_method,
                'plot_image': plot_image,
                'Residual_plot': residual_plot,
            }

            return render(request, 'uploaderApp/all_result.html', context)

        return render(request, 'uploaderApp/results.html', {'method': method,
                                                            'result': result,
                                                            'result1': result1,
                                                            'plot_image': plot_image})

    else:
        form = UploadFileForm()
    return render(request, 'uploaderApp/features_result.html', {'form': form})
