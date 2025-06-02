# Système de Collecte de Données pour la Reconnaissance de Langue des Signes

## Description du Projet

Ce projet implémente un système de collecte de données pour la reconnaissance de langue des signes utilisant MediaPipe et OpenCV. Il permet de capturer et d'analyser les mouvements corporels, faciaux et des mains pour créer des datasets d'entraînement pour des modèles d'apprentissage automatique.

## Architecture du Projet

### Module Utils (`utils.py`)

Le module `utils.py` contient les fonctions utilitaires essentielles pour le traitement des données MediaPipe :

#### Fonctions Principales

- **`mediapipe_detection(image, model)`** : Effectue la détection MediaPipe sur une image
  - Convertit l'image de BGR vers RGB
  - Traite l'image avec le modèle MediaPipe
  - Retourne l'image traitée et les résultats de détection

- **`draw_landmarks(image, results)`** : Dessine les points de repère basiques
  - Affiche les connexions faciales (FACEMESH_TESSELATION)
  - Affiche les connexions de pose corporelle
  - Affiche les connexions des mains gauche et droite

- **`draw_styled_landmarks(image, results)`** : Dessine les points de repère avec un style personnalisé
  - Utilise des couleurs différentes pour chaque type de landmark
  - Applique des épaisseurs et rayons personnalisés pour une meilleure visualisation

- **`extract_keypoints(results)`** : Extrait les coordonnées des points clés
  - **Pose** : 33 points × 4 coordonnées (x, y, z, visibility) = 132 valeurs
  - **Visage** : 468 points × 3 coordonnées (x, y, z) = 1404 valeurs
  - **Main gauche** : 21 points × 3 coordonnées = 63 valeurs
  - **Main droite** : 21 points × 3 coordonnées = 63 valeurs
  - **Total** : 1662 valeurs par frame

### Module Collection (`collection.py`)

Le module `collection.py` implémente le système de collecte de données interactif :

#### Fonctionnalités

- **Collecte Interactive** : Utilisation de la barre d'espace pour démarrer/arrêter l'enregistrement
- **Séquences Temporelles** : Capture de 30 frames par séquence par défaut
- **Gestion Automatique** : Organisation automatique des données par mot et séquence
- **Visualisation en Temps Réel** : Affichage des landmarks pendant la capture
- **Sauvegarde Structurée** : Stockage des données au format NumPy (.npy)

## Installation et Configuration

### Prérequis

- Python 3.10
- Camera fonctionnelle
- Système d'exploitation compatible avec MediaPipe

### Installation des Dépendances

```bash
pip install -r requirements.txt
```

### Dépendances Principales

- **OpenCV** : Traitement d'images et capture vidéo
- **MediaPipe** : Détection et suivi des landmarks corporels
- **NumPy** : Manipulation des données numériques
- **TensorFlow/Keras** : Framework d'apprentissage automatique

## Guide d'Utilisation

### 1. Démarrage du Mode Collecte

Pour commencer la collecte de données, exécutez :

```bash
python collection.py
```

### 2. Configuration de la Collecte

Lors du démarrage, le système vous demandera :

```
Please input the word you want to capture(e.g. hello):
```

Saisissez le mot ou geste que vous souhaitez enregistrer (par exemple : "bonjour", "merci", "au_revoir").

### 3. Processus de Collecte

#### Interface de Collecte

1. **Fenêtre de Prévisualisation** : Une fenêtre OpenCV s'ouvre montrant :
   - Le flux vidéo en temps réel
   - Les landmarks MediaPipe superposés
   - L'état d'enregistrement actuel

2. **Contrôles Clavier** :
   - **Barre d'espace** : Démarrer/arrêter l'enregistrement d'une séquence
   - **Échap** : Fermer l'application (dans la fenêtre OpenCV)

#### Workflow de Collecte

1. **Positionnement** : Placez-vous devant la caméra
2. **Préparation** : Préparez le geste à enregistrer
3. **Démarrage** : Appuyez sur la barre d'espace pour commencer
4. **Exécution** : Effectuez le geste (30 frames seront capturées)
5. **Arrêt** : Appuyez à nouveau sur la barre d'espace ou attendez la capture automatique
6. **Répétition** : Le processus se répète pour 10 séquences par défaut

### 4. Structure des Données Générées

```
MP_Data/
└── [nom_du_mot]/
    ├── 0/
    │   ├── 0.npy
    │   ├── 1.npy
    │   └── ... (jusqu'à 29.npy)
    ├── 1/
    │   ├── 0.npy
    │   └── ...
    └── ... (jusqu'à 9/)
```

Chaque fichier `.npy` contient un vecteur de 1662 valeurs représentant tous les landmarks d'une frame.

### 5. Paramètres Configurables

Dans `collection.py`, vous pouvez modifier :

```python
SEQUENCE_LENGTH = 30    # Nombre de frames par séquence
CAPTURE_TIMES = 10      # Nombre de séquences à capturer
DATA_PATH = 'MP_Data'   # Répertoire de sauvegarde
```

## Conseils d'Utilisation

### Pour une Collecte Optimale

1. **Éclairage** : Assurez-vous d'avoir un bon éclairage
2. **Arrière-plan** : Utilisez un arrière-plan contrasté et uniforme
3. **Position** : Restez à une distance appropriée de la caméra (1-2 mètres)
4. **Stabilité** : Évitez les mouvements brusques de la caméra
5. **Cohérence** : Maintenez une vitesse d'exécution constante pour chaque geste

### Gestion des Erreurs

- **Caméra non détectée** : Vérifiez que votre webcam est connectée et fonctionnelle
- **Landmarks non détectés** : Améliorez l'éclairage ou ajustez votre position
- **Erreurs de sauvegarde** : Vérifiez les permissions d'écriture dans le répertoire

## Développement Futur

Ce système de collecte peut être étendu pour :

- Intégration avec des modèles de classification en temps réel
- Support de gestes plus complexes
- Interface graphique améliorée
- Collecte de données multi-utilisateurs
- Validation automatique de la qualité des données

## Support Technique

Pour toute question ou problème technique, merci de commniquer avec Chen YANG sur WhatsAPP ou Teams

---

# Hand Sign Language Recognition - Preprocessing Module

This project implements a comprehensive preprocessing pipeline for hand sign language recognition using MediaPipe and deep learning.

## Project Structure and Working Directory

**IMPORTANT**: Always run scripts from the **project root directory** to avoid path issues!

```
project_root/                     ← Run all scripts from here
├── src/
│   ├── __init__.py
│   ├── classification/
│   │   └── classifier.py          # LSTM with attention classifier
│   └── preprocessing/             # (ToDo)
│       ├── __init__.py
│       ├── data_preprocessor.py   
│       ├── data_augmentor.py      
│       ├── feature_engineer.py   
│       └── preprocessing_pipeline.py  
├── utils.py                       
├── collection.py                  
├── demo_preprocessing.py          
├── check_project_setup.py         
├── requirements.txt               
├── MP_data/                       
└── processed_data/                
```

### Path Resolution

The preprocessing pipeline automatically resolves all paths relative to the project root directory by:
1. Finding the directory containing `utils.py` (project root marker)
2. Converting relative paths to absolute paths based on project root
3. Logging resolved paths for verification

## Quick Setup Verification

🛠️ **Before starting, verify your setup**:

```bash
# Check if you're in the right directory and everything is set up correctly
python check_project_setup.py
```

This script will:
- Check if you're in the project root directory
- Verify all required files and directories exist
- Test imports to ensure everything works
- Provide guidance if something is wrong

## Data Structure

The system works with MediaPipe hand sign language data:
- **Sequence Length**: 30 frames
- **Feature Dimension**: 1662 features per frame
  - Pose: 132 features (33 points × 4: x,y,z,visibility)
  - Face: 1404 features (468 points × 3: x,y,z)
  - Left Hand: 63 features (21 points × 3: x,y,z)
  - Right Hand: 63 features (21 points × 3: x,y,z)

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Verify setup:
```bash
python check_project_setup.py
```

3. Make sure you have collected data using `collection.py` first:
```bash
python collection.py
```

## Usage

### Quick Start - Demo

Run the preprocessing demo to see how the modules work:

```bash
# Make sure you're in the project root directory
python demo_preprocessing.py
```

### Working Directory Guidelines

 **Correct way**:
```bash
# From project root
python check_project_setup.py
python demo_preprocessing.py
python collection.py
```

**Wrong way**:
```bash
# From subdirectories
cd src/preprocessing/
python ../../demo_preprocessing.py  # This will cause path issues
```

### Using Individual Components

#### 1. Data Preprocessor

```python
from src.preprocessing import DataPreprocessor

config = {
    "sequence_length": 30,
    "feature_dim": 1662,
    "enable_normalization": True,
    "normalization_method": "standard",
    "enable_smoothing": True,
    "enable_interpolation": True,
    "max_zero_ratio": 0.3
}

preprocessor = DataPreprocessor(config)

# Load and process data for a specific word
sequences, info = preprocessor.load_sequence_data("MP_data", "hello")

# Clean individual sequences
cleaned_sequence = preprocessor.clean_sequence(sequence)

# Process entire dataset
processed_data = preprocessor.process_dataset("MP_data", ["hello", "world"])
```

#### 2. Data Augmentor

```python
from src.preprocessing import DataAugmentor

config = {
    "enable_augmentation": True,
    "augmentation_probability": 0.5,
    "enable_noise": True,
    "noise_std": 0.01,
    "enable_time_warping": True,
    "enable_spatial_transform": True
}

augmentor = DataAugmentor(config)

# Augment single sequence
augmented_sequence = augmentor.augment_sequence(sequence)

# Augment entire dataset
augmented_X, augmented_y = augmentor.augment_dataset(X_train, y_train, augmentation_factor=2)
```

#### 3. Feature Engineer

```python
from src.preprocessing import FeatureEngineer

config = {
    "enable_feature_engineering": True,
    "extract_velocity": True,
    "extract_acceleration": True,
    "extract_angles": True,
    "extract_distances": True
}

feature_engineer = FeatureEngineer(config)

# Extract advanced features from sequence
enhanced_sequence = feature_engineer.extract_features(sequence)
```

#### 4. Complete Pipeline

```python
from src.preprocessing import PreprocessingPipeline

config = {
    "raw_data_path": "MP_data",           # Relative to project root
    "processed_data_path": "processed_data",  # Relative to project root
    "validation_split": 0.2,
    "test_split": 0.1,
    "preprocessing": {
        "enable_normalization": True,
        "normalization_method": "standard"
    },
    "augmentation": {
        "enable_augmentation": True,
        "augmentation_factor": 2
    },
    "feature_engineering": {
        "enable_feature_engineering": False
    }
}

pipeline = PreprocessingPipeline(config)

# Process complete dataset
words = ["hello", "world", "thank", "you"]
result = pipeline.process_full_pipeline(words, save_processed=True)

# Access processed data
X_train, y_train = result["X_train"], result["y_train"]
X_val, y_val = result["X_val"], result["y_val"]
X_test, y_test = result["X_test"], result["y_test"]
```

## Todo List

The preprocessing modules contain function signatures and documentation and you have to implement the following:

### DataPreprocessor (src/preprocessing/data_preprocessor.py)

- [ ] `load_sequence_data()`: Load sequences from file system
- [ ] `validate_sequence()`: Check data quality and detect issues
- [ ] `_detect_outliers()`: Implement outlier detection using z-score
- [ ] `clean_sequence()`: Clean data by handling missing values
- [ ] `_interpolate_missing_values()`: Fill missing values using interpolation
- [ ] `_smooth_sequence()`: Apply smoothing filters to reduce noise
- [ ] `normalize_sequences()`: Normalize data using StandardScaler or MinMaxScaler
- [ ] `process_dataset()`: Complete dataset processing workflow
- [ ] `_save_processed_data()`: Save processed data to disk

### DataAugmentor (src/preprocessing/data_augmentor.py)

- [ ] `augment_sequence()`: Apply augmentation to single sequence
- [ ] `_add_noise()`: Add Gaussian noise to data
- [ ] `_time_warping()`: Implement time warping transformation
- [ ] `_magnitude_warping()`: Apply magnitude warping
- [ ] `_window_slicing()`: Implement window slicing technique
- [ ] `_spatial_transformation()`: Apply spatial transformations (rotation, scale, translation)
- [ ] `_speed_variation()`: Change sequence speed
- [ ] `augment_dataset()`: Augment entire dataset

### FeatureEngineer (src/preprocessing/feature_engineer.py)

- [ ] `extract_features()`: Main feature extraction pipeline
- [ ] `_extract_velocity_features()`: Calculate velocity from position
- [ ] `_extract_acceleration_features()`: Calculate acceleration from velocity
- [ ] `_extract_angle_features()`: Calculate angles between key points
- [ ] `_extract_distance_features()`: Calculate distances between landmarks
- [ ] `_extract_relative_position_features()`: Calculate relative positions
- [ ] `_extract_statistical_features()`: Extract statistical features over windows
- [ ] `_extract_hand_shape_features()`: Extract hand shape descriptors
- [ ] `_extract_hand_orientation_features()`: Calculate hand orientations
- [ ] `_extract_pose_angle_features()`: Extract pose angles
- [ ] `_extract_hand_coordinates()`: Extract hand coordinates from sequence
- [ ] `_calculate_angle()`: Calculate angle between three points

### PreprocessingPipeline (src/preprocessing/preprocessing_pipeline.py)

- [ ] `process_full_pipeline()`: Complete preprocessing workflow
- [ ] `_split_dataset()`: Split data into train/validation/test sets
- [ ] `_compute_final_statistics()`: Calculate dataset statistics
- [ ] `_save_processed_dataset()`: Save complete processed dataset
- [ ] `load_processed_dataset()`: Load previously processed data
- [ ] `preprocess_single_sequence()`: Preprocess single sequence for inference

## Integration with Classifier

After implementing the preprocessing modules, integrate with the existing classifier:

```python
from src.preprocessing import PreprocessingPipeline
from src.classification.classifier import AttentionLSTMClassifier

# Process data
pipeline = PreprocessingPipeline(config)
processed_data = pipeline.process_full_pipeline(words)

# Train classifier
classifier_config = {
    "num_classes": len(words),
    "sequence_length": 30,
    "feature_dim": processed_data["X_train"].shape[2]  # May be enhanced by feature engineering
}

classifier = AttentionLSTMClassifier(classifier_config)
classifier.build_model()

history = classifier.train(
    processed_data["X_train"], processed_data["y_train"],
    processed_data["X_val"], processed_data["y_val"]
)
```

## Data Quality Issues to Address

The current dataset has some quality issues that preprocessing should handle:

1. **Zero Values**: ~3.79% of values are zeros (missing detections)
2. **Data Range**: Values range from -1.18 to 2.58
3. **Missing Frames**: Some sequences may have incomplete data
4. **Noise**: Raw MediaPipe data contains detection noise

The preprocessing pipeline is designed to address these issues through:
- Missing value interpolation
- Outlier detection and handling
- Data smoothing and filtering
- Quality validation and reporting

## Tips for Implementation

1. **Start Simple**: Begin with basic implementations and gradually add complexity
2. **Test Incrementally**: Test each function individually before integrating
3. **Handle Edge Cases**: Consider sequences with all zeros, single frames, etc.
4. **Preserve Data Shape**: Ensure outputs maintain expected dimensions
5. **Log Progress**: Use the provided logger to track processing steps
6. **Validate Results**: Check that processed data improves model performance

## Common Issues and Solutions

1. **Path Errors**: 
   - Always run scripts from project root directory
   - Use the setup checker: `python check_project_setup.py`
   - Don't run from subdirectories

2. **Import Errors**: 
   - Make sure the `src` directory is in your Python path
   - Run from project root directory
   - Verify setup with `python check_project_setup.py`

3. **Missing Data**: 
   - Ensure you've collected data using `collection.py` first
   - Check that `MP_data/` directory exists in project root

4. **Memory Issues**: 
   - Process data in batches for large datasets

5. **Shape Mismatches**: 
   - Carefully track array dimensions throughout pipeline

## Development Workflow

```bash
# 1. Verify setup
python check_project_setup.py

# 2. Collect data first (if not done)
python collection.py

# 3. Test preprocessing demo
python demo_preprocessing.py

# 4. Implement preprocessing functions
# Edit files in src/preprocessing/

# 5. Test individual components
python demo_preprocessing.py

# 6. Integrate with classifier
# Use both preprocessing and classification modules
```

Remember: **Always work from the project root directory!** 