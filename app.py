from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import joblib
import numpy as np

from PIL import Image
from keras.models import load_model

class_names=['airplane','automobile','bird','cat','deer',
             'dog','frog','horse','ship','truck']

cifar10_model = load_model('final_model.keras')

app = Flask(__name__)
CORS(app)

@app.route('/cifar10', methods=['POST'])
def cifar10():
    image_upload = request.files["image"]
    img = Image.open(image_upload).resize((32,32))
    img = np.array(img)
    img = img.reshape(1,32,32,3)
    img = img.astype('float32')
    img = img/255
    output = cifar10_model.predict(img)
    np.argmax(output)

    # Return prediction as JSON
    return class_names[int(np.argmax(output))]

@app.route('/iris', methods=['POST'])
def iris():
    model = joblib.load('iris.model')
    req = request.values.get('param')
    inputs = np.array(req.split(','),dtype=np.float32).reshape(1,-1)
    predict_target = model.predict(inputs)
    if predict_target == 0:
        return 'Setosa'
    elif predict_target == 1:
        return 'Versicolour'
    else:
        return 'Virginica'

@app.route('/')
def helloworld():
    return 'Hello World'

@app.route('/area', methods=['GET'])
def area():
    w = float(request.values['w'])
    h = float(request.values['h'])
    return str(w*h)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8081, debug=False)
