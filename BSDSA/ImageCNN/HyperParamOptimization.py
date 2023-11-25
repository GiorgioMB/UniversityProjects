import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix,f1_score, precision_score, recall_score, log_loss
import os
from sklearn.model_selection import train_test_split
from kaggle.api.kaggle_api_extended import KaggleApi
import tensorflow as tf
from tensorflow.keras.layers import Rescaling, Conv2D, MaxPooling2D, Flatten, Dense
from hyperopt import fmin, tpe, hp



# Set parameters
batch_size = 8 #16 #8
img_height = 150
img_width = 150
num_classes = 6
num_folds = 5
##Run only once
#os.environ['KAGGLE_CONFIG_DIR'] = '/kaggle.json'
#dataset_name = 'puneet6060/intel-image-classification'
#api = KaggleApi()
#api.authenticate()
#print("Authenticated, downloading dataset...")
#api.dataset_download_files(dataset_name, path='/Users/micalettog/Desktop/Computer Science/Projects/BSDSA/ImageCNN/dataset/', unzip=True)
#print("Dataset downloaded")
##


dataset_test_dir = r"/Users/micalettog/Desktop/Computer Science/Projects/BSDSA/ImageCNN/dataset/seg_test"
dataset_train_dir = r"/Users/micalettog/Desktop/Computer Science/Projects/BSDSA/ImageCNN/dataset/seg_train"
train_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_train_dir,
    seed=341,
    image_size=(img_height, img_width),
    batch_size=batch_size
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_test_dir,
    seed=341,
    image_size=(img_height, img_width),
    batch_size=batch_size
)
val_true_labels = []  # List to store true labels

for images, labels in val_ds:
    val_true_labels.extend(labels.numpy())

def objective(params):
    model = tf.keras.Sequential([
        Rescaling(1./255),
        Conv2D(params['num_filters'], params['kernel_size'], activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(params['num_filters'], params['kernel_size'], activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(params['num_filters'], params['kernel_size'], activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(params['num_filters'], params['kernel_size'], activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, kernel_regularizer=tf.keras.regularizers.l1(params['l1_lambda']), activation='relu'),
        Dense(64, kernel_regularizer=tf.keras.regularizers.l1(params['l1_lambda']), activation='relu'),
        Dense(32, kernel_regularizer=tf.keras.regularizers.l1(params['l1_lambda']), activation='relu'),
        Dense(6)
    ])
    
    # Compile the model with the specified learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'])
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

    # Train the model (you should replace this with your data loading and training logic)
    history = model.fit(
        train_ds,
        epochs=16,
        validation_data=val_ds,
        verbose=1
    )
    
    predicted_probabilities = model.predict(val_ds)

    val_loss = log_loss(val_true_labels, predicted_probabilities)
    return val_loss

# Define the hyperparameter search space
space = {
    'num_filters': hp.choice('num_filters', [16, 32, 64]),
    'kernel_size': hp.choice('kernel_size', [(3, 3), (5, 5), (7, 7)]),
    'learning_rate': hp.loguniform('learning_rate', -4, -2),
    'l1_lambda': hp.loguniform('l1_lambda', -5, -1)
}

best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=50)
best_params = {
    'num_filters': [16, 32, 64][best['num_filters']],
    'kernel_size': [(3, 3), (5, 5), (7, 7)][best['kernel_size']],
    'learning_rate': best['learning_rate'],
    'l1_lambda': best['l1_lambda']
}

# num_filters: 16 - 64 step: 16
# kernel_size: 3, 4, 5
#learning_rate: e-2, e-4 step: e-2 - e-4 / 300
##TOFIND best
model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(best_params['num_filters'], best_params['kernel_size'], activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(best_params['num_filters'], best_params['kernel_size'], activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(best_params['num_filters'], best_params['kernel_size'], activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(best_params['num_filters'], best_params['kernel_size'], activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    #tf.keras.layers.Conv2D(params['num_filters'], params['kernel_size'], activation='relu'),
    #tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, kernel_regularizer=tf.keras.regularizers.l1(best_params['l1_lambda']), activation='relu'),
    tf.keras.layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l1(best_params['l1_lambda']), activation='relu'),
    tf.keras.layers.Dense(32, kernel_regularizer=tf.keras.regularizers.l1(best_params['l1_lambda']), activation='relu'),
    #tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(6)
])
model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=best_params['learning_rate']),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
