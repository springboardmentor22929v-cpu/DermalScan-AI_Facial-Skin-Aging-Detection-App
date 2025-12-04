# File: download_weights.py (located in project root)

import tensorflow as tf
import os
from tensorflow.keras.applications import EfficientNetB0
import shutil
import sys

print('Attempting to download weights via Keras Applications...')

try:
    # This line triggers Keras to download the weights to its cache
    # It attempts to create the model with 'imagenet' weights, which will trigger download if not cached.
    # Note: This might still fail with 403 Forbidden if Keras's internal download mechanism is also blocked.
    _ = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    print('Download triggered.')

    keras_home = os.path.expanduser('~') + '/.keras'
    # Keras might save the file with one of these two names
    cache_path_old = os.path.join(keras_home, 'models', 'efficientnetb0_weights_tf_dim_ordering_tf_kernels_notop.h5')
    cache_path_new = os.path.join(keras_home, 'models', 'efficientnetb0_notop.h5')

    # Check which path the file might be at
    final_cache_path = None
    if os.path.exists(cache_path_new):
        final_cache_path = cache_path_new
    elif os.path.exists(cache_path_old):
        final_cache_path = cache_path_old

    if final_cache_path:
        print(f'Found weights in Keras cache: {final_cache_path}')
        
        project_models_dir = r'C:\Users\dhana\OneDrive\Desktop\dermalscan-project\models'
        target_path = os.path.join(project_models_dir, 'efficientnetb0_base_notop.weights.h5')
        
        os.makedirs(project_models_dir, exist_ok=True)
        
        shutil.copy2(final_cache_path, target_path)
        print(f'Successfully copied weights to your project: {target_path}')
    else:
        print('Weights not found in Keras cache after trigger. Download might have failed or been blocked.')
        sys.exit("Error: Keras weights not found in cache. This script requires a successful Keras internal download.")

except Exception as e:
    print(f'Download attempt failed: {e}')
    sys.exit(f"Error: Keras internal download failed: {e}. Please try manual download (refer to project README).")