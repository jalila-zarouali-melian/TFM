import os
import logging
from flask import Flask, request, render_template, redirect, url_for, session
import pandas as pd
import plotly.graph_objects as go

app = Flask(__name__, static_folder='static')
app.secret_key = 'your_secret_key'

# Ruta de inicio, redirecciona a la página principal
@app.route('/')
def index():
    return redirect(url_for('main'))

# Ruta para la página principal (Inicio)
@app.route('/main')
def main():
    return render_template('index.html')

# Ruta para la página de carga de archivo
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            logging.info("Nombre del archivo enviado: %s", file.filename)
            filename = file.filename
            file_path = os.path.join('uploads', filename)
            file.save(file_path)
            logging.info("Archivo guardado en: %s", file_path)

            # Almacenar el nombre del archivo en la sesión
            session['filename'] = filename

            return redirect(url_for('process_file'))
    return render_template('upload.html')

# Ruta para procesar el archivo y mostrar las estadísticas
@app.route('/process')
def process_file():
    try:
        filename = session.get('filename')

        if filename is None:
            return redirect(url_for('upload_file'))

        file_path = os.path.join('uploads', filename)
        logging.info("Procesando archivo: %s", file_path)

        # Leer el archivo CSV o Excel utilizando pandas
        if filename.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif filename.endswith('.xls') or filename.endswith('.xlsx'):
            df = pd.read_excel(file_path, engine='xlrd')
        else:
            return "Formato de archivo no compatible. Debe ser un archivo CSV o Excel."

        # Obtener estadísticas adicionales
        null_counts = df.isnull().sum().to_list()
        missing_counts = df.isna().sum().to_list()
        info = df.info().replace('\n', '<br>') if df.info() else None

        # Realizar el "shape" y "describe" del DataFrame
        shape_info = df.shape
        logging.info("Shape del DataFrame: %s", shape_info)
        describe_info = df.describe().to_html()
        logging.info("Describe del DataFrame:\n%s", describe_info)

        return render_template('stats.html', filename=filename, shape_info=shape_info, describe_info=describe_info, null_counts=null_counts, missing_counts=missing_counts, info=info)

    except Exception as e:
        return render_template('error.html', error_message=str(e))

# Ruta para la página del gráfico
@app.route('/graph')
def show_graph():
    try:
        filename = session.get('filename')

        if filename is None:
            return redirect(url_for('upload_file'))

        file_path = os.path.join('uploads', filename)
        logging.info("Procesando archivo: %s", file_path)

        # Leer el archivo CSV o Excel utilizando pandas
        if filename.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif filename.endswith('.xls') or filename.endswith('.xlsx'):
            df = pd.read_excel(file_path, engine='xlrd')
        else:
            return "Formato de archivo no compatible. Debe ser un archivo CSV o Excel."

        # Crear gráfico utilizando Plotly
        fig = go.Figure(data=[go.Bar(x=df.columns, y=df.count())])
        graph = fig.to_html(full_html=False)

        return render_template('graph.html', graph=graph)

    except Exception as e:
        return render_template('error.html', error_message=str(e))

if __name__ == '__main__':
    app.run(debug=True)
