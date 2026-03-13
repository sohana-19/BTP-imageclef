import os
from knn import KNN

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input

# ======================================================
# CHANGE PATHS HERE
# ======================================================

knn_configuration = {

    'data': {

        # -----------------------------
        # CHANGE THESE IMAGE PATHS
        # -----------------------------
        'train_images_folder': 'ROCov2/images/train/train/',
        'val_images_folder': 'ROCov2/images/val/valid/',
        'test_images_folder': 'ROCov2/images/test/test/',

        # -----------------------------
        # CHANGE THESE CSV PATHS
        # -----------------------------
        'train_data_path': 'ROCov2/annotations/train_concepts.csv',
        'val_data_path': 'ROCov2/annotations/valid_concepts.csv',
        'test_data_path': 'ROCov2/annotations/test_concepts.csv',

        'skip_head': True,
        'split_token': ',',
        'img_size': (224, 224, 3),
    },

    'encoder': {
        'backbone': DenseNet121(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        ),
        'preprocessor': preprocess_input,
    },

    'knn_params': {
        'k_range': (15, 55),
        'k_step': 12,
        'r_range': (1, 6),
        'weighted': False
    },

    'embeddings': {
        'load_embeddings': True
    },

    'save_results': True,
    'results_path': 'knn_results.csv'
}

k = KNN(configuration=knn_configuration)
k.run()
