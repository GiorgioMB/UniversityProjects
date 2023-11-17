import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix,f1_score, precision_score, recall_score
from keras.models import Model
from tensorflow.keras.layers.experimental import preprocessing
import os
from sklearn.model_selection import train_test_split

# Set parameters
batch_size = 8 #16 #8
img_height = 150
img_width = 150
num_classes = 6
num_folds = 5
dataset_dir = r"C:\Users\Stefano\Desktop\Stefano\Universita_Bocconi\2_Magistrale\Associazione\Image_Classification_with_CNN\data\seg_train\seg_train" ##TOCHANGE

# Use os.path.normpath to handle the path properly
dataset_dir = os.path.normpath(dataset_dir)

dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size,
)


train_ds = tf.keras.utils.image_dataset_from_directory(
  r"C:\Users\Stefano\Desktop\Stefano\Universita_Bocconi\2_Magistrale\Associazione\Image_Classification_with_CNN\data\seg_train\seg_train", ##TOCHANGE
  #validation_split=0.3,
  #subset="training",
  seed=341,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  r"C:\Users\Stefano\Desktop\Stefano\Universita_Bocconi\2_Magistrale\Associazione\Image_Classification_with_CNN\data\seg_test\seg_test", ##TOCHANGE
  #validation_split=0.3,
  #subset="validation",
  seed=341,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_f1 = []
val_precision = []
val_recall = []

all_train_acc = []
all_val_acc = []
all_train_loss = []
all_val_loss = []
all_conf_matrix = []
best_params = {'num_filters': 16, 'kernel_size': 3, 'learning_rate': 0.0005}

# K-Fold Cross-Validation loop
for train_index, val_index in kf.split(images):
    train_x, val_x = images[train_index], images[val_index]
    train_y, val_y = labels[train_index], labels[val_index]

    train_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y)).batch(batch_size)
    val_ds = tf.data.Dataset.from_tensor_slices((val_x, val_y)).batch(batch_size)

    # Define the model architecture
    model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(best_params['num_filters'], best_params['kernel_size']),
    tf.keras.layers.PReLU(),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(best_params['num_filters'], best_params['kernel_size']),
    tf.keras.layers.PReLU(),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128),
    tf.keras.layers.PReLU(),
    tf.keras.layers.Dense(5)
    ])

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=best_params['learning_rate']),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    #apply data augmentation
    #train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))


    # Train the model
    history = model.fit(
        train_ds,
        epochs=16,
        validation_data=val_ds,
        verbose=1
    )

    # Save metrics
    all_train_acc.append(history.history['accuracy'])
    all_val_acc.append(history.history['val_accuracy'])
    all_train_loss.append(history.history['loss'])
    all_val_loss.append(history.history['val_loss'])

    # Generate confusion matrix for this fold and save
    y_pred = np.argmax(model.predict(val_ds), axis=1)
    y_true = [y for _, y in val_ds]
    y_true = np.concatenate(y_true)
    conf_matrix = confusion_matrix(y_true, y_pred)
    all_conf_matrix.append(conf_matrix)

    f1 = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')

    val_f1.append(f1)
    val_precision.append(precision)
    val_recall.append(recall)

    print(f"Fold {len(val_f1)}: F1 Score = {f1}, Precision = {precision}, Recall = {recall}")
