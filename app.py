import os
import logging
from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
import plotly.graph_objects as go

app = Flask(__name__)

# Ruta para la página principal
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            logging.info("Nombre del archivo enviado: %s", file.filename)
            filename = file.filename
            file_path = os.path.join('uploads', filename)
            file.save(file_path)
            logging.info("Archivo guardado en: %s", file_path)
            return redirect(url_for('process_file', filename=filename))
    return render_template('index.html')

# Ruta para procesar el archivo
@app.route('/process/<filename>')
def process_file(filename):
    try:
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

        # Crear gráfico utilizando Plotly
        fig = go.Figure(data=[go.Bar(x=df.columns, y=df.count())])
        graph = fig.to_html(full_html=False)

        return render_template('index.html', filename=filename, shape_info=shape_info, describe_info=describe_info, null_counts=null_counts, missing_counts=missing_counts, info=info, graph=graph)

    except Exception as e:
        logging.error("Error al procesar el archivo: %s", str(e))
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)
