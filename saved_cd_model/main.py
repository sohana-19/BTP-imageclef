import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input

from knn import KNN

# ======================================================
# CONFIGURATION FOR IMAGECLEF 2026
# ======================================================

knn_configuration = {
    'data': {
        # Path to images folder (contains all images)
        'images_folder': 'dev_concept/images/',  # Update this path
        
        # Path to concepts CSV file
        'concepts_path': 'dev_concept/concepts.csv',  # Update this path
        
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
    
    'embeddings': {
        # Set to False to extract embeddings fresh
        'load_embeddings': False
    }
}

# ======================================================
# VERIFY FILES EXIST
# ======================================================

def verify_files():
    """Check if required files exist"""
    print("="*60)
    print("IMAGECLEF 2026 - CONCEPT DETECTION")
    print("="*60)
    
    issues = []
    
    # Check images folder
    if not os.path.exists(knn_configuration['data']['images_folder']):
        issues.append(f"Images folder not found: {knn_configuration['data']['images_folder']}")
    else:
        images = [f for f in os.listdir(knn_configuration['data']['images_folder']) 
                 if f.endswith('.jpg')]
        print(f"✓ Images folder found: {len(images)} images")
        
        train_images = [f for f in images if 'train' in f]
        valid_images = [f for f in images if 'valid' in f]
        print(f"  - Training images: {len(train_images)}")
        print(f"  - Validation images: {len(valid_images)}")
    
    # Check concepts file
    if not os.path.exists(knn_configuration['data']['concepts_path']):
        issues.append(f"Concepts file not found: {knn_configuration['data']['concepts_path']}")
    else:
        import pandas as pd
        df = pd.read_csv(knn_configuration['data']['concepts_path'])
        print(f"✓ Concepts file found: {len(df)} entries")
        print(f"  Sample ID: {df['ID'].iloc[0]}")
    
    if issues:
        print("\n❌ ERROR: The following issues were found:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nPlease update the paths in the configuration above.")
        return False
    
    return True

# ======================================================
# SCRIPT TO LOAD MODEL AND RUN INFERENCE ON OFFICIAL TEST
# ======================================================

def run_inference_on_official_test():
    """
    This function will be used when the official test set is released
    """
    print("\n" + "="*60)
    print("LOADING TRAINED MODEL FOR OFFICIAL TEST SET")
    print("="*60)
    
    # Initialize KNN
    knn = KNN(configuration=knn_configuration)
    
    # Load the trained model
    if knn.load_model("knn_model.pkl"):
        # Run inference on official test set
        test_images_folder = "official_test/images/"  # Update this path
        predictions = knn.inference_on_test_set(test_images_folder, "submission_test.csv")
        
        print("\n" + "="*60)
        print("INFERENCE COMPLETE")
        print("="*60)
        print(f"Predictions saved to: submission_test.csv")
        print("You can now submit this file to the competition")
        print("="*60)
    else:
        print("Please train the model first by running: python main.py")

# ======================================================
# MAIN
# ======================================================

if __name__ == "__main__":
    # Verify files
    if not verify_files():
        sys.exit(1)
    
    print("\n" + "="*60)
    print("Running KNN Training with New Split Strategy")
    print("="*60)
    print("Data Split Strategy:")
    print("  1. Combine all original train and valid images")
    print("  2. Split into 80% new train, 20% new validation")
    print("  3. Original validation set becomes test set")
    print("  4. Model will be saved for future inference")
    print("="*60)
    
    # Initialize KNN
    knn = KNN(configuration=knn_configuration)
    
    # Run the pipeline (training + validation + testing)
    final_score = knn.run()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Final F1 Score: {final_score:.4f}")
    print("\nModel saved as: knn_model.pkl")
    print("\nWhen official test set is released, run:")
    print("  python -c 'from main import run_inference_on_official_test; run_inference_on_official_test()'")
    print("="*60)

