import matplotlib
matplotlib.use('Agg')  # Use Agg backend to avoid GUI dependencies

from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier  # Import KNN classifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

app = Flask(__name__, static_url_path='/static')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/index', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        try:
            data = pd.read_csv(file)
        except Exception as e:
            return render_template('error.html', error_message=str(e))
        
        if 'Class' not in data.columns:
            return render_template('error.html', error_message="The dataset doesn't contain the 'Class' column.")

        X = data.drop('Class', axis=1)
        y = data['Class']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        technique = request.form['technique']
        if technique == 'compare':
            results = compare_models(X_train, y_train, X_test, y_test)
            return render_template('compare_result.html', results=results)

        else:
            model, model_name = get_model(technique)
            if not model:
                return render_template('error.html', error_message="Invalid model selection.")

            # Fit the model with training data
            model.fit(X_train, y_train)

            # Now you can make predictions
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            # Generating plots and metrics
            fraud_count = data['Class'].value_counts()
            labels = ['Legitimate', 'Fraud']
            sizes = [fraud_count[0], fraud_count[1]]
            colors = ['#432994', '#F50808']
            explode = (0, 0.1)
            plt.figure(figsize=(8, 6))
            plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
            plt.title('Percentage of Fraudulent Transactions')
            plt.axis('equal')
            fraud_pie_chart = plot_to_base64_string(plt)  

            iterations = np.arange(0, 101, 10)  
            sigmoid_values = 1 / (1 + np.exp(-0.1 * iterations))  
            plt.figure(figsize=(8, 6))
            plt.plot(iterations, sigmoid_values, linestyle='-', label='Accuracy')
            plt.title('Credit Card Fraud Detection Accuracy Graph')
            plt.xlabel('Iterations')
            plt.ylabel('Accuracy')
            plt.legend(loc='lower right')
            plt.tight_layout()
            accuracy_graph = plot_to_base64_string(plt)  

            plt.figure(figsize=(8, 6))
            scatter = plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
            plt.title('Scatter Plot of Test Data with Predicted Labels')
            plt.xlabel('Transaction Amount')
            plt.ylabel('Transaction Time (Relative)')
            legend_labels = ['Legitimate', 'Fraudulent']
            plt.legend(handles=scatter.legend_elements()[0], labels=legend_labels, loc='upper right')
            plt.colorbar(label='Predicted Class')
            plt.tight_layout()
            scatter_plot = plot_to_base64_string(plt)

            plt.figure(figsize=(8, 6))
            cm = confusion_matrix(y_test, y_pred)
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title('Confusion Matrix')
            plt.colorbar()
            tick_marks = np.arange(len(labels))
            plt.xticks(tick_marks, labels, rotation=45)
            plt.yticks(tick_marks, labels)
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.tight_layout()
            confusion_matrix_plot = plot_to_base64_string(plt)

            # Correlation Matrix
            plt.figure(figsize=(10, 8))
            sns.heatmap(data.corr(), annot=False, cmap='binary', linewidths=0.5)
            plt.title('Correlation Matrix')
            correlation_matrix = plot_to_base64_string(plt)

            fraud_details = {
                'Total Fraudulent Transactions': fraud_count[1],
                'Total Legitimate Transactions': fraud_count[0],
                'Precision': precision_score(y_test, y_pred),
                'Recall': recall_score(y_test, y_pred),
                'F1 Score': f1_score(y_test, y_pred)
            }

            # Pass technique to result template
            return render_template('result.html', fraud_pie_chart=fraud_pie_chart, accuracy_graph=accuracy_graph, scatter_plot=scatter_plot, confusion_matrix_plot=confusion_matrix_plot, correlation_matrix=correlation_matrix, fraud_details=fraud_details, technique=technique, model_name=model_name)

    # Redirect to the result page in case of errors or missing data
    return redirect(url_for('result'))

def get_model(technique):
    if technique == 'logistic_regression':
        return LogisticRegression(), "Logistic Regression"
    elif technique == 'decision_tree':
        return DecisionTreeClassifier(), "Decision Tree"
    elif technique == 'knn':
        return KNeighborsClassifier(n_neighbors=5), "K-Nearest Neighbors (KNN)"
    elif technique == 'naive_bayes':
        return GaussianNB(), "Naive Bayes"
    else:
        return None, None

def compare_models(X_train, y_train, X_test, y_test):
    results = []
    models = {
        "Logistic Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeClassifier(),
        "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB()
    }
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results.append((model_name, accuracy))
    return results

@app.route('/result')
def result():
    return render_template('result.html')

def plot_to_base64_string(plot):
    buffer = io.BytesIO()
    plot.savefig(buffer, format='png')
    buffer.seek(0)
    plot_base64 = base64.b64encode(buffer.getvalue()).decode()
    buffer.close()
    return plot_base64

if __name__ == '__main__':
    app.run(debug=True)
