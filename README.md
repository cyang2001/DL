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