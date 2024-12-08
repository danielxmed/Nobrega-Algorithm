# Nobrega's Algorithm

**Nobrega's Algorithm** is a robust, deep learning-based image classification pipeline optimized for medical imaging tasks. It leverages Bayesian hyperparameter optimization to find the best architecture and training setup for multi-class medical image classification. This approach enables healthcare professionals and researchers to systematically improve their models and achieve higher validation accuracy while minimizing overfitting.

## Key Features

1. **Directory-Based Dataset Structure**:  
   The algorithm expects a structured directory format:
   - `dataset/train/<class_1>`, `dataset/train/<class_2>`, ..., `dataset/train/<class_n>`
   - `dataset/validation/<class_1>`, `dataset/validation/<class_2>`, ..., `dataset/validation/<class_n>`
   
   Each folder represents a distinct class and contains the corresponding training or validation images.

2. **Automatic Class Detection**:  
   The algorithm automatically identifies the number of classes based on the subdirectories within the `train` and `validation` folders. This ensures scalability as you add or remove classes.

3. **Preprocessing and Data Augmentation**:  
   Data is normalized and optionally augmented (rotation, zoom, flips, brightness adjustments) to enhance generalization, especially when dealing with limited medical datasets.

4. **Flexible Architecture Selection**:  
   The algorithm can:
   - Build a custom Convolutional Neural Network (CNN) architecture from scratch.
   - Perform transfer learning using popular pretrained models such as ResNet50, InceptionV3, EfficientNet, MobileNet, DenseNet, or VGG16.  
   
   It also tunes parameters like the number of layers, filters, activation functions, normalization layers, dense layers, dropout rates, and more.

5. **Bayesian Hyperparameter Optimization**:  
   Utilizing Bayesian optimization, the algorithm intelligently explores the hyperparameter space. Instead of random or grid search, it uses prior information from completed trials to guide subsequent trials, converging on optimal configurations more efficiently.

6. **Adaptive Trial Management**:  
   - Starts with a baseline of 100 trials.  
   - Continues or stops the search based on improvement in validation accuracy (≥0.1% improvements over certain intervals, with customizable thresholds).  
   - Can add incremental sets of 50 additional trials if improvements continue.  
   - Has a hard limit at 500 trials, extendable only if significant accuracy improvements are detected.
   
   This adaptive mechanism avoids unnecessary computations when results plateau and allows extended exploration when meaningful improvements are still possible.

7. **Overfitting Check**:  
   After each trial, the algorithm checks for overfitting. Trials that show a large discrepancy between training and validation accuracy (e.g., ≥10% difference) are deprioritized in the final selection.

8. **Final Model Selection and Export**:  
   After completing the search, Nobrega's Algorithm selects the best hyperparameter set that yields high validation accuracy without severe overfitting. It then retrains the model using these optimal hyperparameters and saves the final `.keras` model file.

9. **Prediction Capability**:  
   Alongside `train.py`, a `predict.py` script allows you to load the final model and classify new, unseen images. The same preprocessing pipeline ensures consistency between training and inference.

## Requirements

- Python 3.9+  
- TensorFlow/Keras (version 2.6+ recommended)  
- Keras Tuner for Bayesian optimization (`pip install keras-tuner`)  
- NumPy, Pillow, and other standard packages (e.g., `os`, `json`)  
- GPU support is highly recommended for faster training.

## How to Use

1. **Prepare Your Dataset**:  
   Organize your images as follows:  
   ```
   dataset/
   ├─ train/
   │  ├─ class_1/
   │  ├─ class_2/
   │  └─ ...
   └─ validation/
      ├─ class_1/
      ├─ class_2/
      └─ ...
   ```
   
2. **Train the Model with Bayesian Optimization**:  
   Run:
   ```bash
   python train.py
   ```
   The algorithm will:
   - Identify classes
   - Begin the Bayesian search for the optimal hyperparameters
   - Dynamically continue or halt search trials based on performance criteria
   - Eventually train the final model with the best configuration
   - Save the final model to `best_model.keras` and the selected hyperparameters to `best_hparams.json`

   With `verbose=1` in `train.py`, you will see epoch-by-epoch training logs, including training and validation metrics.

3. **Predict on New Images**:  
   Once the final model is saved, run:
   ```bash
   python predict.py path_to_image.jpg
   ```
   
   This will:
   - Load `best_model.keras` and its associated preprocessing steps
   - Predict the class with the highest probability
   - Print the predicted class name to the console

## Customization

- **Adjusting Hyperparameters**:  
  You can modify the hyperparameter search space directly in `train.py` within the `build_model` function and the `run_trial` logic.  
- **Data Augmentation**:  
  Tweak augmentation parameters (rotation, zoom, brightness, flips) in the `get_datasets` function.  
- **Early Stopping and Performance Criteria**:  
  Adjust early stopping patience or thresholds for saturation and improvement in `train.py`.

## Intended Use and Limitations

Nobrega's Algorithm is designed for medical image classification tasks, such as classifying various pathologies in medical scans. It follows best practices but should be considered as a starting point. For real-world clinical use, it may require:

- Additional regulatory compliance checks.
- More extensive validation and calibration on representative medical data.
- Integration into a broader medical decision support system.

**Disclaimer**: The provided code and models are intended for research and educational purposes only. They are not meant to replace professional medical advice, diagnosis, or treatment.

## Contributing

Contributions are welcome! You can open issues on GitHub for improvements, bug reports, or feature requests. Feel free to fork and submit pull requests.

## License

This project can be distributed under an open-source license of your choice (e.g., MIT, Apache 2.0). 

