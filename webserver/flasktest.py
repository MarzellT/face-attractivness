import keras
from io import BytesIO
import tensorflow as tf
import numpy as np
import base64
from PIL import Image
from flask import Flask, render_template, redirect, request
from os import path
import sys
sys.path.append('../')
import attractivness

model = attractivness.create_model()
model.load_weights('../logs/151020192133/epoch07.hdf5')
model._make_predict_function()
graph = tf.get_default_graph()

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def home():
    global graph
    
    if request.method == 'GET':
        return render_template("index.html")

    file = request.files.get('file')

    if not file:
        return render_template('index.html')

    if file:
        input_file = file.stream
    else:
        return render_template('index.html')

    encodings, locations = attractivness.prepare_image(file)
    with graph.as_default():
        pred = model.test_on_batch(encodings)
    output = attractivness.visualize_result(file, pred*9+1, fontsize=round(
        Image.open(file).size[0]/8))

    img = Image.open(file)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((256,256))
    output.thumbnail((500,500))

    with img as img:
        buffer = BytesIO()
        img.save(buffer, 'JPEG')
        img = buffer.getvalue()
        input_image = base64.b64encode(img).decode()
    with output as output:
        buffer = BytesIO()
        output.save(buffer, 'JPEG')
        output = buffer.getvalue()
        output = base64.b64encode(output).decode()
    
    return render_template('index.html', input_image=input_image, result_image=output,
            prediction=pred)
    
if __name__ == "__main__":
    app.run(debug=False)
