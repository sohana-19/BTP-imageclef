import os
import numpy as np
from tqdm import tqdm
from collections import Counter
import tensorflow as tf
import utilities as utils


class KNN:

    def __init__(self, configuration):

        self.configuration = configuration

        self.train_images_folder = configuration['data']['train_images_folder']
        self.val_images_folder = configuration['data']['val_images_folder']
        self.test_images_folder = configuration['data']['test_images_folder']

        self.train_data_path = configuration['data']['train_data_path']
        self.val_data_path = configuration['data']['val_data_path']
        self.test_data_path = configuration['data']['test_data_path']

        self.img_size = configuration['data']['img_size']

        self.k_range = configuration['knn_params']['k_range']
        self.k_step = configuration['knn_params']['k_step']
        self.r_range = configuration['knn_params']['r_range']

        self.train_data = {}
        self.val_data = {}
        self.test_data = {}

        self.tags_list = []

        self.training_embeddings = {}
        self.validation_embeddings = {}
        self.test_embeddings = {}

        self.encoder = None
        self.ids = None


    # =====================================================
    # INITIALIZE DATA STRUCTURES
    # =====================================================

    def init_structures(self, skip_head=False, split_token=','):

        self.train_data = self.load_csv_data(self.train_data_path, skip_head, split_token)
        self.val_data = self.load_csv_data(self.val_data_path, skip_head, split_token)
        self.test_data = self.load_csv_data(self.test_data_path, skip_head, split_token)

        print("Training samples:", len(self.train_data))
        print("Validation samples:", len(self.val_data))
        print("Test samples:", len(self.test_data))

        self.tags_list = self.load_tags(self.train_data)

        print("Total concepts:", len(self.tags_list))


    # =====================================================
    # LOAD CSV DATA
    # =====================================================

    def load_csv_data(self, file_name, skip_head=False, split_token=','):

        data = {}

        with open(file_name, 'r') as f:

            if skip_head:
                next(f)

            for line in f:

                parts = line.strip().split(split_token)

                image = parts[0] + ".jpg"
                concepts = parts[1]

                data[image] = concepts

        print("Loaded:", file_name)

        return data


    # =====================================================
    # LOAD TAGS
    # =====================================================

    def load_tags(self, training_data):

        tags = []

        for img in training_data:
            tags.extend(training_data[img].split(';'))

        return list(set(tags))


    # =====================================================
    # LOAD ENCODER
    # =====================================================

    def load_encoder(self, model):

        self.encoder = tf.keras.Model(
            inputs=model.input,
            outputs=model.output
        )

        print(self.encoder.summary())


    # =====================================================
    # SAVE / LOAD EMBEDDINGS
    # =====================================================

    def save_embeddings(self, embeddings, filename):

        np.save(filename, embeddings)
        print("Saved embeddings:", filename)


    def load_embeddings(self, filename):

        embeddings = np.load(filename, allow_pickle=True).item()
        print("Loaded embeddings:", filename)
        return embeddings


    # =====================================================
    # EXTRACT EMBEDDINGS
    # =====================================================

    def get_embeddings(self, data, images_path):

        embeddings = {}

        for img_name in tqdm(data):

            path = os.path.join(images_path, img_name)

            img = tf.keras.preprocessing.image.load_img(
                path,
                target_size=self.img_size[:2]
            )

            img_array = tf.keras.preprocessing.image.img_to_array(img)

            img_array = np.expand_dims(img_array, axis=0)

            img_array = self.configuration['encoder']['preprocessor'](img_array)

            embedding = self.encoder.predict(img_array, verbose=0).flatten()

            embedding = embedding / np.linalg.norm(embedding)

            embeddings[img_name] = embedding

        return embeddings


    # =====================================================
    # TRAIN / TEST PIPELINE
    # =====================================================

    def train_tune_test(self):

        self.init_structures(
            skip_head=self.configuration['data']['skip_head'],
            split_token=self.configuration['data']['split_token']
        )

        self.load_encoder(self.configuration['encoder']['backbone'])

        load_flag = self.configuration['embeddings']['load_embeddings']

        # ============================================
        # TRAIN EMBEDDINGS
        # ============================================

        if load_flag and os.path.exists("train_embeddings.npy"):

            self.training_embeddings = self.load_embeddings("train_embeddings.npy")

        else:

            print("Extracting training embeddings...")
            self.training_embeddings = self.get_embeddings(
                self.train_data,
                self.train_images_folder
            )

            self.save_embeddings(self.training_embeddings, "train_embeddings.npy")

        # ============================================
        # VALIDATION EMBEDDINGS
        # ============================================

        if load_flag and os.path.exists("val_embeddings.npy"):

            self.validation_embeddings = self.load_embeddings("val_embeddings.npy")

        else:

            print("Extracting validation embeddings...")
            self.validation_embeddings = self.get_embeddings(
                self.val_data,
                self.val_images_folder
            )

            self.save_embeddings(self.validation_embeddings, "val_embeddings.npy")

        # ============================================
        # TEST EMBEDDINGS
        # ============================================

        if load_flag and os.path.exists("test_embeddings.npy"):

            self.test_embeddings = self.load_embeddings("test_embeddings.npy")

        else:

            print("Extracting test embeddings...")
            self.test_embeddings = self.get_embeddings(
                self.test_data,
                self.test_images_folder
            )

            self.save_embeddings(self.test_embeddings, "test_embeddings.npy")

        # ============================================

        self.ids = list(self.training_embeddings.keys())

        training_embeddings_array = np.array(
            [self.training_embeddings[i] for i in self.ids]
        )

        best_k = self.k_range[1]
        best_r = self.r_range[1]

        test_score, test_results = self.test(best_k, best_r, training_embeddings_array)

        print("Final F1:", test_score)

        return test_score, test_results


    # =====================================================
    # TEST FUNCTION
    # =====================================================

    def test(self, best_k, best_r, training_embeddings_array):

        y_pred_test = {}

        for test_img in tqdm(self.test_data):

            sims = training_embeddings_array @ self.test_embeddings[test_img]

            top_k = np.argsort(sims)[-best_k:]

            tags_list = []

            for index in top_k:

                tags = self.train_data[self.ids[index]].split(';')

                tags_list.extend(tags)

            most_frequent_tags = Counter(tags_list).most_common(best_r)

            y_pred_test[test_img] = ';'.join(t[0] for t in most_frequent_tags)

        f1_score = utils.evaluate_f1(self.test_data, y_pred_test)

        return f1_score, y_pred_test


    # =====================================================
    # RUN
    # =====================================================

    def run(self):

        test_score, test_results = self.train_tune_test()

        if self.configuration['save_results']:

            with open(self.configuration['results_path'], 'w') as out_test:

                for result in test_results:

                    out_test.write(result + ',' + test_results[result] + '\n')
