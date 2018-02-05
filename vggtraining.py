from keras.engine import  Model
from keras.layers import Flatten, Dense, Input, Dropout
from keras_vggface.vggface import VGGFace
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, LambdaCallback, CSVLogger, TensorBoard
from keras.utils import np_utils
from keras.optimizers import SGD
import requests
import sys

# Informar por argumento o numero de classes que serão treinadas
nb_class = int(sys.argv[1])

#nb_class = 12
hidden_dim = 4096

# Controle  
#urlProcess = 'http://localhost:8880/api/recognitionconfig/statusprocessing/'
#urlComplete = 'http://localhost:8880/api/recognitionconfig/statuscomplete/'
#urlIndicesPost = 'http://localhost:8880/api/model/reg/'

try:
    reply = requests.get(urlProcess)
#    requests.post(url, files=files)
    print('processando..!!1!', nb_class)
except Exception as e:
    print(e)

# Diretorio de checkpoint
checkpoint_path = './tmp/weights.hdf5'

vgg_model = VGGFace(include_top=False, input_shape=(224,224, 3), weights=None)
vgg_model.load_weights('./model/rcmalli_vggface_tf_notop_v2.h5')

# Desabilita as camadas da CNN
for layer in vgg_model.layers:
    layer.trainable = False

last_layer = vgg_model.get_layer('pool5').output
flat = Flatten(name='flatten')(last_layer)
d1 = Dense(hidden_dim, activation='relu', name='fc6')(flat)
drop1 = Dropout(0.5)(d1)
d2 = Dense(hidden_dim, activation='relu', name='fc7')(drop1)
drop2 = Dropout(0.5)(d2)
out = Dense(nb_class, activation='softmax', name='fc8')(drop2)
model = Model(inputs=vgg_model.input, outputs=out)
#Carrega checkpoint
#model.load_weights(checkpoint_path)
model.summary()


x_train = './data/train'
x_test = './data/validation'

datagen = ImageDataGenerator(rescale=1./224,
				featurewise_center=False,             # set input mean to 0 over the dataset
                samplewise_center=False,              # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                samplewise_std_normalization=False,   # divide each input by its std
                zca_whitening=False,                  # apply ZCA whitening
                rotation_range=10,                     # randomly rotate images in the range (degrees, 0 to 180)
                width_shift_range=0.2,                # randomly shift images horizontally (fraction of total width)
                height_shift_range=0.2,               # randomly shift images vertically (fraction of total height)
                horizontal_flip=True,                 # randomly flip images
                vertical_flip=False) 

datagenTest = ImageDataGenerator(rescale=1./224,
				featurewise_center=False,             # set input mean to 0 over the dataset
                samplewise_center=False,              # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                samplewise_std_normalization=False,   # divide each input by its std
                zca_whitening=False,                  # apply ZCA whitening
                rotation_range=10,                     # randomly rotate images in the range (degrees, 0 to 180)
                width_shift_range=0.2,                # randomly shift images horizontally (fraction of total width)
                height_shift_range=0.2,               # randomly shift images vertically (fraction of total height)
                horizontal_flip=True,                 # randomly flip images
                vertical_flip=False)

#datagen.fit(x_train#)
#datagenTest.fit(x_test)

img_width = 224
img_height = 224

epochs=100
steps_per_epoch=16
batch_size=32

train_generator = datagen.flow_from_directory(
    x_train,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    #class_mode='categorical'
)

test_generator = datagenTest.flow_from_directory(
    x_test,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

print(str(train_generator.class_indices))
indices = str(train_generator.class_indices)
#print(indices)
try:
    reply = requests.post(urlIndicesPost, data=indices, headers={'Content-Type': 'text/plain'})
#    requests.post(url, files=files)
    print('processando..!!1!', nb_class)
except Exception as e:
    print(e)


#


# print(test_generator.target_size)

opt_sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=opt_sgd, loss='mean_squared_error', metrics=['accuracy'])
#model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
checkpointer = ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=True, mode='auto', period=1)
logging_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: print('epoch: {}    loss:{}    acc:{}'.format(epoch, logs['loss'], logs['acc'])))
csv_logger = CSVLogger('training_ramon_final.log')
tbCallBack = TensorBoard(log_dir='./tensorboard_ramon_final', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)

#print("Trainning")
# Se necessario treinar primeiro as ultimas camadas
#model.fit_generator(train_generator,
##                                 use_multiprocessing=True,
#				 steps_per_epoch=steps_per_epoch,
#                                 epochs=epochs,
#                                 validation_data=test_generator, validation_steps=100, callbacks=[checkpointer, logging_callback, csv_logger, tbCallBack])


# Habilitar caso necessário treinar todas as camadas(treinando apenas as ultimas com os pesos da VGGFace já é suficiente)
#for layer in model.layers:
#    layer.trainable = True

model.compile(optimizer=opt_sgd, loss='mean_squared_error', metrics=['accuracy'])

print("Fine Tunning")
model.fit_generator(train_generator,
                                 use_multiprocessing=True,
                                 steps_per_epoch=steps_per_epoch,
                                 epochs=epochs,
                                 validation_data=test_generator, validation_steps=100, callbacks=[checkpointer, logging_callback, csv_logger, tbCallBack])

model.save('cnn_pre_trained_result.h5')
print('model salvo')
try:
    reply = requests.get(urlComplete)
    print('concluido!')
except Exception as e:
    print(e)