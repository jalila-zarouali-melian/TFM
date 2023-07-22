import os
from flask import Flask, request, render_template, redirect, url_for
import pandas as pd

app = Flask(_name_)


# Ruta para la página principal
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            # Crear el directorio "uploads" si no existe
            if not os.path.exists('uploads'):
                os.makedirs('uploads')

            filename = file.filename
            file.save(os.path.join('uploads', filename))
            return redirect(url_for('process_file', filename=filename))
    return render_template('index.html')


# Ruta para procesar el archivo
@app.route('/process/<filename>')
def process_file(filename):
    try:
        file_path = os.path.join('uploads', filename)

        # Leer el archivo CSV o Excel utilizando pandas
        if filename.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif filename.endswith('.xls') or filename.endswith('.xlsx'):
            df = pd.read_excel(file_path, engine='xlrd')
        else:
            return "Formato de archivo no compatible. Debe ser un archivo CSV o Excel."

        # Realizar el "shape" y "describe" del DataFrame
        shape_info = df.shape
        describe_info = df.describe().to_html()

        return render_template('index.html', filename=filename, shape_info=shape_info, describe_info=describe_info)

    except Exception as e:
        return str(e)


if _name_ == '_main_':
    app.run(debug=True)