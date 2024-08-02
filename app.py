from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Ensure the uploads directory exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Custom filter to make zip available in Jinja2 templates
@app.template_filter('zip')
def zip_filter(a, b):
    return zip(a, b)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' in request.files and request.files['file'].filename != '':
            # Handle file upload
            file = request.files['file']
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            data = pd.read_csv(file_path)
        elif 'data' in request.form and request.form['data'] != '':
            # Handle manual data entry
            from io import StringIO
            data = pd.read_csv(StringIO(request.form['data']))

        # Redirect to results page with data
        return redirect(url_for('results', data=data.to_json()))

    return render_template('index.html')

@app.route('/results')
def results():
    # Get data from request
    data_json = request.args.get('data')
    data = pd.read_json(data_json)

    # Preprocess the data (example preprocessing)
    data['datetime'] = pd.to_datetime(data['datetime'])
    data['hour'] = data['datetime'].dt.hour
    data['day'] = data['datetime'].dt.dayofweek
    data['month'] = data['datetime'].dt.month
    X = data[['hour', 'day', 'month']]
    y = data['coffee_name']

    # Split the data and train the model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Pass results to template
    return render_template('results.html', accuracy=accuracy, y_test=y_test.tolist(), y_pred=y_pred.tolist())

if __name__ == '__main__':
    app.run(debug=True)
