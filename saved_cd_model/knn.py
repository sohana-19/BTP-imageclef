import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
import utilities as utils
import optuna
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
import re
import pickle
import joblib


class KNN:

    def __init__(self, configuration):

        self.configuration = configuration

        self.images_folder = configuration['data']['images_folder']
        self.concepts_path = configuration['data']['concepts_path']
        self.img_size = configuration['data']['img_size']

        # Data structures
        self.all_data = {}  # image_id -> concepts (CUIs)
        self.train_data = {}
        self.val_data = {}
        self.test_data = {}

        self.training_embeddings = {}
        self.validation_embeddings = {}
        self.test_embeddings = {}

        self.encoder = None
        self.unique_cuis = None
        self.tag_to_idx = None
        self.train_labels_full = None
        self.idf_weights = None
        self.train_ids_list = None
        self.best_params = None
        self.train_matrix = None  # For faster inference


    # ================= DATA LOADING =================

    def load_concepts(self):
        """Load concepts from CSV file"""
        print("\n" + "="*50)
        print("Loading Concepts Data")
        print("="*50)
        
        df = pd.read_csv(self.concepts_path)
        concepts_dict = {}
        
        for _, row in df.iterrows():
            image_id = row['ID']
            cuis = row['CUIs']
            concepts_dict[image_id] = cuis
        
        print(f"Loaded concepts for {len(concepts_dict)} images")
        return concepts_dict


    def split_by_filename(self):
        """
        Split data based on filename patterns:
        - Train: contains 'train' in ID
        - Validation: contains 'valid' in ID
        """
        print("\n" + "="*50)
        print("Splitting Data by Filename Patterns")
        print("="*50)
        
        train_ids = []
        val_ids = []
        
        for img_id in self.all_data.keys():
            if 'train' in img_id.lower():
                train_ids.append(img_id)
            elif 'valid' in img_id.lower() or 'val' in img_id.lower():
                val_ids.append(img_id)
        
        print(f"Original training images: {len(train_ids)}")
        print(f"Original validation images: {len(val_ids)}")
        
        return train_ids, val_ids


    def create_new_splits(self, original_train_ids, original_val_ids):
        """
        Combine all data and split 80:20 for new train/val
        Original validation becomes test set
        """
        print("\n" + "="*50)
        print("Creating New Train/Val/Test Splits")
        print("="*50)
        
        # Combine all available data for training
        all_available_ids = original_train_ids + original_val_ids
        print(f"Total available images for training: {len(all_available_ids)}")
        
        # Split into new train (80%) and new validation (20%)
        new_train_ids, new_val_ids = train_test_split(
            all_available_ids, 
            test_size=0.2, 
            random_state=42,
            stratify=None
        )
        
        # Original validation becomes test set
        test_ids = original_val_ids
        
        print(f"\nNew Training images: {len(new_train_ids)}")
        print(f"New Validation images: {len(new_val_ids)}")
        print(f"Test images (original validation): {len(test_ids)}")
        
        return new_train_ids, new_val_ids, test_ids


    def init_structures(self):
        """Initialize data structures with new splits"""
        
        # Load all concepts
        self.all_data = self.load_concepts()
        
        # Split by filename to separate train/valid
        original_train_ids, original_val_ids = self.split_by_filename()
        
        # Create new splits: combine all, split 80:20, original valid becomes test
        new_train_ids, new_val_ids, test_ids = self.create_new_splits(
            original_train_ids, original_val_ids
        )
        
        # Create data dictionaries
        self.train_data = {img_id: self.all_data[img_id] for img_id in new_train_ids if img_id in self.all_data}
        self.val_data = {img_id: self.all_data[img_id] for img_id in new_val_ids if img_id in self.all_data}
        self.test_data = {img_id: self.all_data[img_id] for img_id in test_ids if img_id in self.all_data}
        
        print("\n" + "="*50)
        print("Final Dataset Statistics:")
        print("="*50)
        print(f"Training images: {len(self.train_data)}")
        print(f"Validation images: {len(self.val_data)}")
        print(f"Test images (original validation): {len(self.test_data)}")
        
        # Display sample IDs
        if len(self.train_data) > 0:
            print(f"\nSample train ID: {list(self.train_data.keys())[0]}")
        if len(self.val_data) > 0:
            print(f"Sample validation ID: {list(self.val_data.keys())[0]}")
        if len(self.test_data) > 0:
            print(f"Sample test ID: {list(self.test_data.keys())[0]}")
        
        self.build_label_matrix()
        self.compute_cui_frequencies()


    def build_label_matrix(self):
        """Convert CUI concepts to multi-hot encoding"""
        
        print("\n" + "="*50)
        print("Building Label Matrix")
        print("="*50)
        
        # Collect all unique CUIs from training data only
        all_cuis = set()
        for concepts in self.train_data.values():
            cuis = concepts.split(';')
            all_cuis.update(cuis)
        
        self.unique_cuis = sorted(list(all_cuis))
        self.tag_to_idx = {cui: idx for idx, cui in enumerate(self.unique_cuis)}
        print(f"Total unique CUIs: {len(self.unique_cuis)}")
        
        # Build training label matrix
        self.train_labels_full = np.zeros((len(self.train_data), len(self.unique_cuis)), dtype=np.float32)
        self.train_ids_list = list(self.train_data.keys())
        
        for i, (img_id, concepts) in enumerate(self.train_data.items()):
            cuis = concepts.split(';')
            for cui in cuis:
                if cui in self.tag_to_idx:
                    self.train_labels_full[i, self.tag_to_idx[cui]] = 1.0
        
        # Print label statistics
        avg_cuis = np.mean(np.sum(self.train_labels_full, axis=1))
        print(f"Average CUIs per training image: {avg_cuis:.2f}")
        print(f"Min CUIs per image: {np.min(np.sum(self.train_labels_full, axis=1))}")
        print(f"Max CUIs per image: {np.max(np.sum(self.train_labels_full, axis=1))}")


    def compute_cui_frequencies(self):
        """Compute inverse frequency weights for CUIs (IDF weighting)"""
        
        print("\n" + "="*50)
        print("Computing IDF Weights")
        print("="*50)
        
        tag_counts = np.sum(self.train_labels_full, axis=0)
        tag_frequencies = tag_counts / len(self.train_data)
        
        # IDF weighting with smoothing
        self.idf_weights = 1.0 / (tag_frequencies + 0.01)
        self.idf_weights = self.idf_weights / np.mean(self.idf_weights)
        
        # Print weight statistics
        print(f"IDF weight range: {np.min(self.idf_weights):.2f} - {np.max(self.idf_weights):.2f}")
        print(f"Mean IDF weight: {np.mean(self.idf_weights):.2f}")


    # ================= ENCODER =================

    def load_encoder(self, model):
        """Load pre-trained encoder model"""
        self.encoder = tf.keras.Model(
            inputs=model.input,
            outputs=model.output
        )
        print("\n" + "="*50)
        print("Encoder Loaded")
        print("="*50)
        print(f"Input shape: {model.input.shape}")
        print(f"Output shape: {model.output.shape}")


    # ================= EMBEDDINGS =================

    def save_embeddings(self, embeddings, filename):
        """Save embeddings to disk"""
        np.save(filename, embeddings)
        print(f"Saved: {filename}")


    def load_embeddings(self, filename):
        """Load embeddings from disk"""
        embeddings = np.load(filename, allow_pickle=True).item()
        print(f"Loaded: {filename}")
        return embeddings


    def get_embeddings(self, data, batch_size=128):
        """Extract embeddings for images with batch processing"""
        embeddings = {}
        img_ids = list(data.keys())
        
        if len(img_ids) == 0:
            print("No images to process")
            return embeddings
        
        print(f"\nExtracting embeddings for {len(img_ids)} images...")
        
        for i in tqdm(range(0, len(img_ids), batch_size), desc="Extracting embeddings"):
            batch_ids = img_ids[i:i+batch_size]
            batch_images = []
            valid_batch_ids = []
            
            for img_id in batch_ids:
                # Image path: images_folder/ID.jpg
                path = os.path.join(self.images_folder, f"{img_id}.jpg")
                
                try:
                    img = tf.keras.preprocessing.image.load_img(
                        path,
                        target_size=self.img_size[:2]
                    )
                    img_array = tf.keras.preprocessing.image.img_to_array(img)
                    batch_images.append(img_array)
                    valid_batch_ids.append(img_id)
                except Exception as e:
                    print(f"Warning: Could not load {path}: {e}")
                    continue
            
            if not batch_images:
                continue
                
            batch_images = np.array(batch_images)
            batch_images = self.configuration['encoder']['preprocessor'](batch_images)
            batch_embeddings = self.encoder.predict(batch_images, verbose=0)
            
            for img_id, embedding in zip(valid_batch_ids, batch_embeddings):
                embedding = embedding.flatten()
                # L2 normalization for cosine similarity
                embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
                embeddings[img_id] = embedding
        
        print(f"Successfully extracted embeddings for {len(embeddings)} images")
        return embeddings


    # ================= MODEL SAVING =================

    def save_model(self, filename="knn_model.pkl"):
        """
        Save the trained model including:
        - Best parameters (k, r, use_idf, adaptive_threshold)
        - Training embeddings
        - Label matrix
        - Unique CUIs
        - IDF weights
        - Training IDs
        """
        print("\n" + "="*50)
        print("Saving Model")
        print("="*50)
        
        # Prepare model data
        model_data = {
            'best_params': self.best_params,
            'unique_cuis': self.unique_cuis,
            'tag_to_idx': self.tag_to_idx,
            'idf_weights': self.idf_weights,
            'train_labels_full': self.train_labels_full,
            'training_embeddings': self.training_embeddings,
            'train_ids_list': self.train_ids_list,
            'img_size': self.img_size,
            'configuration': self.configuration
        }
        
        # Save to file
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        # Also save training matrix for faster inference
        if hasattr(self, 'train_matrix') and self.train_matrix is not None:
            np.save("train_matrix.npy", self.train_matrix)
        
        print(f"✓ Model saved to: {filename}")
        print(f"  - Best parameters: {self.best_params}")
        print(f"  - Unique CUIs: {len(self.unique_cuis)}")
        print(f"  - Training samples: {len(self.train_ids_list)}")
        
        return model_data


    def load_model(self, filename="knn_model.pkl"):
        """
        Load a trained model for inference on test set
        """
        print("\n" + "="*50)
        print("Loading Trained Model")
        print("="*50)
        
        if not os.path.exists(filename):
            print(f"❌ Error: Model file not found: {filename}")
            return False
        
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        
        # Load all components
        self.best_params = model_data['best_params']
        self.unique_cuis = model_data['unique_cuis']
        self.tag_to_idx = model_data['tag_to_idx']
        self.idf_weights = model_data['idf_weights']
        self.train_labels_full = model_data['train_labels_full']
        self.training_embeddings = model_data['training_embeddings']
        self.train_ids_list = model_data['train_ids_list']
        
        # Load training matrix if available
        if os.path.exists("train_matrix.npy"):
            self.train_matrix = np.load("train_matrix.npy")
        
        print(f"✓ Model loaded from: {filename}")
        print(f"  - Best parameters: {self.best_params}")
        print(f"  - Unique CUIs: {len(self.unique_cuis)}")
        print(f"  - Training samples: {len(self.train_ids_list)}")
        
        return True


    # ================= ADVANCED PREDICTION =================

    def predict_advanced(self, similarities, k, r, use_idf=True, adaptive_threshold=True, 
                         train_labels=None, idf_weights=None):
        """
        Advanced prediction with multiple techniques:
        - Exponential weighting
        - IDF weighting for rare CUIs
        - Adaptive thresholding based on confidence
        """
        if train_labels is None:
            train_labels = self.train_labels_full
        if idf_weights is None:
            idf_weights = self.idf_weights
            
        # Ensure k is not larger than available neighbors
        if len(similarities) < k:
            k = len(similarities)
        
        # Get top k indices with complete sorting (preserves order)
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        top_k_similarities = similarities[top_k_indices]
        
        # Exponential weighting for better discrimination
        weights = np.exp(top_k_similarities * 2)
        
        # Aggregate labels with weights
        label_scores = np.zeros(len(self.unique_cuis))
        for idx, weight in zip(top_k_indices, weights):
            label_scores += train_labels[idx] * weight
        
        # Apply IDF weighting if enabled
        if use_idf:
            label_scores = label_scores * idf_weights
        
        # Adaptive threshold based on similarity distribution
        if adaptive_threshold and np.max(label_scores) > 0:
            threshold = 0.15 * np.max(label_scores)
            label_scores[label_scores < threshold] = 0
        
        # Select top r CUIs
        top_r_indices = np.argsort(label_scores)[-r:][::-1]
        predicted_cuis = [self.unique_cuis[i] for i in top_r_indices if label_scores[i] > 0]
        
        # Ensure we always have at least one CUI
        if not predicted_cuis and len(top_r_indices) > 0:
            predicted_cuis = [self.unique_cuis[top_r_indices[0]]]
        
        # Remove duplicates
        predicted_cuis = list(dict.fromkeys(predicted_cuis))
        
        return ';'.join(predicted_cuis)


    # ================= INFERENCE ON NEW TEST SET =================

    def inference_on_test_set(self, test_images_folder, output_file="submission_test.csv"):
        """
        Run inference on official test set using saved model
        """
        print("\n" + "="*50)
        print("Running Inference on Official Test Set")
        print("="*50)
        
        # Load encoder
        self.load_encoder(self.configuration['encoder']['backbone'])
        
        # Get test image IDs
        test_ids = []
        if os.path.exists(test_images_folder):
            test_files = [f for f in os.listdir(test_images_folder) if f.endswith('.jpg')]
            test_ids = [f.replace('.jpg', '') for f in test_files]
        
        print(f"Test images found: {len(test_ids)}")
        
        if len(test_ids) == 0:
            print("❌ No test images found!")
            return {}
        
        # Extract test embeddings
        test_data = {img_id: "" for img_id in test_ids}
        test_embeddings = self.get_embeddings(test_data, batch_size=128)
        
        # Prepare matrices
        test_embeddings_list = [test_embeddings[img_id] for img_id in test_ids]
        test_matrix = np.array(test_embeddings_list)
        
        # Prepare training matrix
        train_embeddings_list = [self.training_embeddings[img_id] for img_id in self.train_ids_list]
        train_matrix = np.array(train_embeddings_list)
        
        # Normalize
        test_matrix = normalize(test_matrix, norm='l2')
        train_matrix = normalize(train_matrix, norm='l2')
        
        # Compute similarities
        sim_matrix = np.dot(test_matrix, train_matrix.T)
        
        # Predict
        predictions = {}
        for i, img_id in enumerate(tqdm(test_ids, desc="Predicting")):
            similarities = sim_matrix[i]
            predicted = self.predict_advanced(
                similarities,
                self.best_params["k"],
                self.best_params["r"],
                self.best_params.get("use_idf", True),
                self.best_params.get("adaptive_threshold", True)
            )
            predictions[img_id] = predicted
        
        # Save submission
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("ID,CUIs\n")
            for img_id, cuis in predictions.items():
                f.write(f"{img_id},{cuis}\n")
        
        print(f"\n✓ Submission saved: {output_file}")
        print(f"  Total predictions: {len(predictions)}")
        
        return predictions


    # ================= OPTUNA OPTIMIZATION =================

    def objective_efficient(self, trial, sim_matrix, val_keys, train_labels_subset):
        """Optuna objective for hyperparameter tuning"""
        k = trial.suggest_int("k", 20, 80)
        r = trial.suggest_int("r", 2, 6)
        use_idf = trial.suggest_categorical("use_idf", [True, False])
        adaptive_threshold = trial.suggest_categorical("adaptive_threshold", [True, False])
        
        # Use 30% of validation data for faster tuning
        sample_size = min(3000, len(val_keys))
        np.random.seed(trial.number)
        sampled_indices = np.random.choice(len(val_keys), sample_size, replace=False)
        
        y_pred_val = {}
        for idx in sampled_indices:
            similarities = sim_matrix[idx]
            predicted = self.predict_advanced(
                similarities, k, r, use_idf, adaptive_threshold, 
                train_labels=train_labels_subset
            )
            y_pred_val[val_keys[idx]] = predicted
        
        # Evaluate F1 on sampled validation data
        sampled_val_data = {val_keys[idx]: self.val_data[val_keys[idx]] for idx in sampled_indices}
        score = utils.evaluate_f1(sampled_val_data, y_pred_val)
        
        return score


    # ================= OPTIMIZED TESTING =================

    def test_optimized(self, best_params, train_embeddings, train_labels):
        """Optimized testing with full training set"""
        print("\n" + "="*50)
        print("Running Final Test on Original Validation Set")
        print("="*50)
        
        y_pred_test = {}
        test_keys = list(self.test_data.keys())
        
        if len(test_keys) == 0:
            print("No test images found!")
            return 0.0, {}
        
        # Convert to matrices for efficient computation
        test_embeddings_list = [self.test_embeddings[k] for k in test_keys]
        test_matrix = np.array(test_embeddings_list)
        train_matrix = np.array(list(train_embeddings.values()))
        
        # Store for later inference
        self.train_matrix = train_matrix
        
        # Normalize for cosine similarity
        test_matrix = normalize(test_matrix, norm='l2')
        train_matrix = normalize(train_matrix, norm='l2')
        
        # Compute all similarities at once
        sim_matrix = np.dot(test_matrix, train_matrix.T)
        print(f"Similarity matrix shape: {sim_matrix.shape}")
        
        # Predict in batches
        batch_size = 200
        for i in tqdm(range(0, len(test_keys), batch_size), desc="Testing"):
            batch_indices = range(i, min(i+batch_size, len(test_keys)))
            for idx in batch_indices:
                similarities = sim_matrix[idx]
                predicted = self.predict_advanced(
                    similarities, 
                    best_params["k"], 
                    best_params["r"], 
                    best_params.get("use_idf", True),
                    best_params.get("adaptive_threshold", True),
                    train_labels=train_labels
                )
                y_pred_test[test_keys[idx]] = predicted
        
        # Calculate final F1 score
        f1_score = utils.evaluate_f1(self.test_data, y_pred_test)
        return f1_score, y_pred_test


    # ================= SUBMISSION FORMAT =================

    def save_submission(self, results, output_file="submission.csv"):
        """Save results in competition format"""
        
        print("\n" + "="*50)
        print("Creating Submission File")
        print("="*50)
        
        # Remove duplicate IDs and CUIs
        unique_results = {}
        for img_id, cuis in results.items():
            unique_cuis = ';'.join(sorted(set(cuis.split(';'))))
            unique_results[img_id] = unique_cuis
        
        # Write to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("ID,CUIs")
            for img_id, cuis in unique_results.items():
                f.write(f"\n{img_id},{cuis}")
        
        print(f"✓ Submission file saved: {output_file}")
        print(f"  Total entries: {len(unique_results)}")
        
        return output_file


    # ================= MAIN PIPELINE =================

    def train_tune_test(self):
        """Complete training, tuning, and testing pipeline"""
        
        # Phase 1: Data preparation
        self.init_structures()
        self.load_encoder(self.configuration['encoder']['backbone'])
        
        load_flag = self.configuration['embeddings']['load_embeddings']
        
        # Phase 2: Extract embeddings
        print("\n" + "="*50)
        print("Phase 1: Extracting Embeddings")
        print("="*50)
        
        if load_flag and os.path.exists("train_embeddings_2026.npy"):
            self.training_embeddings = self.load_embeddings("train_embeddings_2026.npy")
        else:
            self.training_embeddings = self.get_embeddings(self.train_data, batch_size=128)
            self.save_embeddings(self.training_embeddings, "train_embeddings_2026.npy")
        
        if load_flag and os.path.exists("val_embeddings_2026.npy"):
            self.validation_embeddings = self.load_embeddings("val_embeddings_2026.npy")
        else:
            self.validation_embeddings = self.get_embeddings(self.val_data, batch_size=128)
            self.save_embeddings(self.validation_embeddings, "val_embeddings_2026.npy")
        
        # Phase 3: Compute similarity matrix
        print("\n" + "="*50)
        print("Phase 2: Computing Similarity Matrix")
        print("="*50)
        
        if len(self.validation_embeddings) == 0:
            print("No validation embeddings! Skipping similarity computation.")
            self.best_params = {"k": 61, "r": 3, "use_idf": False, "adaptive_threshold": True}
        else:
            val_matrix = np.array(list(self.validation_embeddings.values()))
            train_matrix = np.array(list(self.training_embeddings.values()))
            
            # Store training matrix for inference
            self.train_matrix = train_matrix
            
            # Normalize for cosine similarity
            val_matrix = normalize(val_matrix, norm='l2')
            train_matrix = normalize(train_matrix, norm='l2')
            
            # Compute similarity matrix
            sim_matrix = np.dot(val_matrix, train_matrix.T)
            print(f"Similarity matrix shape: {sim_matrix.shape}")
            
            # Phase 4: Hyperparameter optimization with Optuna
            val_keys = list(self.val_data.keys())
            
            print("\n" + "="*50)
            print("Phase 3: Hyperparameter Optimization (Optuna)")
            print("="*50)
            print("Using 3000 validation samples for faster tuning (30% of data)")
            
            study = optuna.create_study(direction="maximize")
            
            study.optimize(
                lambda trial: self.objective_efficient(
                    trial, sim_matrix, val_keys, self.train_labels_full
                ), 
                n_trials=40
            )
            
            self.best_params = study.best_params
            print("\n" + "="*50)
            print("Best Parameters Found:")
            print("="*50)
            for param, value in self.best_params.items():
                print(f"  {param} = {value}")
            print(f"\nBest Validation F1: {study.best_value:.4f}")
        
        # Phase 5: Load test embeddings (original validation set)
        print("\n" + "="*50)
        print("Phase 4: Testing on Original Validation Set")
        print("="*50)
        
        if load_flag and os.path.exists("test_embeddings_2026.npy"):
            self.test_embeddings = self.load_embeddings("test_embeddings_2026.npy")
        else:
            if len(self.test_data) > 0:
                self.test_embeddings = self.get_embeddings(self.test_data, batch_size=128)
                self.save_embeddings(self.test_embeddings, "test_embeddings_2026.npy")
            else:
                print("No test data found!")
                self.test_embeddings = {}
        
        # Phase 6: Final testing
        if len(self.test_embeddings) > 0:
            test_score, test_results = self.test_optimized(
                self.best_params, self.training_embeddings, self.train_labels_full
            )
        else:
            print("\nNo test set found!")
            test_score = 0.0
            test_results = {}
        
        # Phase 7: Save the trained model
        self.save_model("knn_model.pkl")
        
        print("\n" + "="*50)
        print("Final Results")
        print("="*50)
        print(f"Test F1 Score (on original validation set): {test_score:.4f}")
        
        return test_score, test_results


    # ================= RUN =================

    def run(self):
        """Main run method"""
        test_score, test_results = self.train_tune_test()
        
        if len(test_results) > 0:
            # Save submission file
            submission_file = self.save_submission(test_results, "submission.csv")
            
            # Save backup
            backup_file = "knn_results_backup.csv"
            with open(backup_file, 'w', encoding='utf-8') as f:
                for img_id, cuis in test_results.items():
                    f.write(f"{img_id},{cuis}\n")
            
            print(f"\n✓ Backup saved: {backup_file}")
            
            print("\n" + "="*60)
            print("TRAINING COMPLETE - MODEL SAVED")
            print("="*60)
            print("Files generated:")
            print(f"  1. {submission_file} - Ready for evaluation")
            print(f"  2. knn_model.pkl - Trained model (for future inference)")
            print(f"  3. train_matrix.npy - Training embeddings matrix")
            print(f"  4. train_embeddings_2026.npy - Training embeddings")
            print(f"\nF1 Score: {test_score:.4f}")
            print("\n" + "="*60)
            print("WHEN OFFICIAL TEST SET IS RELEASED:")
            print("="*60)
            print("1. Load the model: knn.load_model('knn_model.pkl')")
            print("2. Run inference: knn.inference_on_test_set('path/to/test/images/')")
            print("3. This will generate submission_test.csv")
            print("="*60)
        else:
            print("\nNo results generated!")
        
        return test_score
