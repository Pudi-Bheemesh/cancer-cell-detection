import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, InputLayer, BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import RMSprop

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter



import tensorflow_addons as tfa
import tensorflow as tf
import tensorflow_addons as tfa
img_size = 128
batch_size = 64

"""train_datagen = ImageDataGenerator(
    preprocessing_function=lambda x: median_filter(x,size=(2, 2), mode='constant', cval=0),
    rescale = 1/255.#, brightness_range = [0.5,1.5],rotation_range=20,shear_range=10, zoom_range = 0.2, 
    #width_shift_range = 0.15, height_shift_range = 0.15, horizontal_flip = True 
)"""
train_datagen = ImageDataGenerator(
    #reprocessing_function=lambda x: median_filter(x, size=(2, 2, 3), mode='constant', cval=0),
    rescale = 1./255,brightness_range = [0.5,1.5],rotation_range=0.2,shear_range=0.3, zoom_range = 0.2, 
    width_shift_range = 0.2, height_shift_range = 0.2,
    horizontal_flip = True 
)
"""train_datagen = ImageDataGenerator(
    preprocessing_function=lambda x: median_filter(x,size=(2, 2), mode='constant', cval=0),
    rescale = 1./255,
    horizontal_flip = True)"""

val_datagen = ImageDataGenerator(rescale = 1./255)
test_datagen = ImageDataGenerator(rescale = 1./255.)

train_gen_dir = "D:\Projects\\friends\darshan\mega project\dataset\dataset2-master\dataset2-master\images\TRAIN"
train_gen_dir_backup = "/kaggle/input/blood-cells/dataset2-master/dataset2-master/images/TRAIN"

val_gen_dir = "D:\Projects\\friends\darshan\mega project\dataset\dataset2-master\dataset2-master\images\TEST_SIMPLE"
val_gen_dir_BACKUP = "/kaggle/input/blood-cells/dataset2-master/dataset2-master/images/TEST_SIMPLE"

test_gen_dir = "D:\Projects\\friends\darshan\mega project\dataset\dataset2-master\dataset2-master\images\TEST"
test_gen_dir_backup = "/kaggle/input/blood-cells/dataset2-master/dataset2-master/images/TEST"

train_generator = train_datagen.flow_from_directory(train_gen_dir,
                                                   target_size = (img_size, img_size),
                                                   batch_size = batch_size,
                                                   shuffle=True,
                                                   class_mode='categorical')

val_generator = val_datagen.flow_from_directory(val_gen_dir,
                                                   target_size = (img_size, img_size),
                                                   batch_size = batch_size,
                                                   shuffle=False,
                                                   class_mode='categorical')

test_generator = test_datagen.flow_from_directory(test_gen_dir,
                                                   target_size = (img_size, img_size),
                                                   batch_size = batch_size,
                                                   shuffle=False,
                                                   class_mode = 'categorical')

input_layer = Input(shape=(128, 128, 3))

x = Conv2D(6, 
           kernel_size=(1,1), 
           strides=(1,1), 
           padding='same', 
           activation='relu')(input_layer)

x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.3)(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(16, kernel_size=(5,5), strides=(1,1),padding='same',activation='relu')(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.3)(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(64, kernel_size=(5,5), strides=(1,1),padding='same',activation='relu')(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.3)(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(128, kernel_size=(5,5), strides=(1,1),padding='same',activation='relu')(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.3)(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(512, kernel_size=(4,4), strides=(1,1),padding='same',activation='relu')(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.3)(x)
x = Dropout(0.2)(x)

x = Flatten()(x) 
output_layer = Dense(4, activation='softmax')(x)

model = Model(inputs=input_layer,outputs=output_layer)
model.summary()
print(train_generator.class_indices)
input()


initial_learning_rate = 0.001
decay_steps = 1000
decay_rate = 0.9
grad_decay = 0.1
optimizer = tf.keras.optimizers.Adam(
    learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate
    ),
    beta_1=1.0 - grad_decay  # Set beta_1 to 0.9 for 0.1 gradient decay
)
model.compile(
    optimizer=optimizer,
    loss = 'categorical_crossentropy',
    metrics=['accuracy']
)


from tensorflow.keras.models import load_model

#model = load_model("/kaggle/input/resnet-50/tensorflow2/classification/1")


history = model.fit_generator(
    train_generator,
    validation_data = val_generator,
    epochs = 20
)


history_dict = history.history
train_acc = history_dict['loss']
val_acc = history_dict['val_loss']
epochs = range(1, len(history_dict['loss'])+1)
plt.plot(epochs, train_acc,'b', label='Training error')
plt.plot(epochs, val_acc,'b', color="orange", label='Validation error')
plt.title('Training and Validation error')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.legend()
plt.show()

history_dict = history.history
train_acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
epochs = range(1, len(history_dict['accuracy'])+1)
plt.plot(epochs, train_acc,'b', label='Training accuracy')
plt.plot(epochs, val_acc,'b', color="orange", label='Validation accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()


y_preds = model.predict_generator(test_generator)

test_generator = test_datagen.flow_from_directory(test_gen_dir,
                                                   target_size = (img_size, img_size),
                                                   batch_size = batch_size,
                                                   shuffle=False,
                                                   class_mode = 'categorical')

x, y = test_generator.next()
y_true = y
for i in range(2487//64):
    x, y = test_generator.next()
    y_true = np.concatenate([y_true, y], axis = 0)
    
print(y_true)


y_true = np.argmax(y_true, axis = 1)
y_preds = np.argmax(y_preds, axis = 1)

print(y_true, y_preds)

import seaborn as sns
import matplotlib.pyplot as plt     
from sklearn.metrics import confusion_matrix


cm = confusion_matrix(y_true, y_preds)
ax= plt.subplot()
sns.heatmap(cm, annot=True, fmt='g', ax=ax)

ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL'])
ax.yaxis.set_ticklabels(['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL'])


from sklearn.metrics import classification_report

print(classification_report(y_true, y_preds, target_names = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']))

model.save("./model")