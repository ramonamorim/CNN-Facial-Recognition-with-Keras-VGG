import os
from flask import Flask, request, redirect, url_for, send_from_directory
from werkzeug import secure_filename
import requests
from keras.preprocessing import image
from keras_vggface.vggface import VGGFace
from keras.models import Model, load_model
from keras.optimizers import SGD, RMSprop
import numpy as np
from predict import predict_img
#from flaskext.mysql import MySQL

app = Flask(__name__)

UPLOAD_FOLDER = './uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#MySqlConfig
#app.config['MYSQL_DATABASE_USER'] = 'root'
#app.config['MYSQL_DATABASE_DB'] = 'facial'
#app.config['MYSQL_DATABASE_HOST'] = 'localhost'
#app.config['MYSQL_DATABASE_PASSWORD']='12345'


ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

urlGet = 'http://localhost:8880/api/model/'

url = 'http://localhost:8880/api/model'





@app.route('/', methods=['GET', 'POST'])
def upload_file():
    
    if request.method == 'POST':
        
        file = request.files['files']
        
        if file and allowed_file(file.filename):
        
            print('**found file', file.filename)
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))  

            print('carregando model')
            # Colocar path do resultado do treino
            model = load_model('/mnt/data/cnn/vggpretrained/cnn_pre_trained_result.h5')    
            model.summary()

            result = predict_img(model)
            r =  requests.get(urlGet)
            python_obj  = json.loads(r.text)

            for val in python_obj:
                nome = val.get('nome')
                cod = str(val.get('codigo'))
                if(result == cod):                                                            
                    payload = {
                                "app_key": "XK2uiixRPIbQVKYQFJHh",
                                "app_secret": "w9i5wEN08ajhqtFn7o2SFl9uaBFAyrvrV3axz9gYCpG0pxC1MfENr53Zp9eLTqfJ",
                                "target_type": "app",
                                "content": nome + " esta aos arredores"
                                } 
                else:
                    payload = {
                                    "app_key": "XK2uiixRPIbQVKYQFJHh",
                                    "app_secret": "w9i5wEN08ajhqtFn7o2SFl9uaBFAyrvrV3axz9gYCpG0pxC1MfENr53Zp9eLTqfJ",
                                    "target_type": "app",
                                    "content": "Uma pessoa desconhecida esta aos arredores"
                            } 

            # Envia notificação de push no App
            r = requests.post("https://api.pushed.co/1/push", data=payload)
            
            #r = requests.post("https://api.pushed.co/1/push", data=payload)
            print(r.text)

            del model


            # for browser, add 'redirect' function on top of 'url_for'
            return 'sucesso!'

def allowed_file(filename):
    return filename[-3:].lower() in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)    
    




