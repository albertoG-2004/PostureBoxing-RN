# model_trainer.py
import tensorflow as tf
import numpy as np
import os
from PIL import Image
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

def load_data(folder_path):
    """
    Carga imágenes a color desde carpetas de clases (Cruzado, Gancho derecha, etc.) directamente
    en la carpeta principal. Redimensiona las imágenes a un tamaño mayor (ej: 128x128).

    Args:
        folder_path (str): Ruta a la carpeta principal del dataset (ej: 'dataset').

    Returns:
        tuple: (images_data, labels_data, class_labels_list)
               - images_data (numpy.ndarray): Array con los datos de las imágenes redimensionadas y normalizadas (sin aplanar).
               - labels_data (numpy.ndarray): Array con las etiquetas numéricas correspondientes a las clases.
               - class_labels_list (list): Lista de nombres de las clases (etiquetas originales).
    """
    images_data = []
    labels_data = []
    class_labels_list = []
    target_size = (128, 128)  # Aumentamos la resolución a 128x128

    try:
        class_folders = [f.path for f in os.scandir(folder_path) if f.is_dir()]
        if not class_folders:
            raise ValueError("La carpeta debe contener subcarpetas con las clases de boxeo (Cruzado, Gancho derecha, etc.).")

        class_labels_set = set()

        for class_folder in class_folders:
            class_name = os.path.basename(class_folder)
            class_labels_set.add(class_name)
            for filename in os.listdir(class_folder):
                if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    continue
                image_path = os.path.join(class_folder, filename)
                try:
                    img = Image.open(image_path) # Ya no convertimos a escala de grises, mantenemos color
                    img = img.resize(target_size, Image.LANCZOS)
                    img_array = np.array(img) / 255.0 # Normalizamos los valores de color (0-1)
                    images_data.append(img_array)
                    labels_data.append(class_name)
                except Exception as e:
                    print(f"Error al cargar imagen {filename}: {e}")

        if images_data:
            images_data = np.array(images_data)
            label_encoder = LabelEncoder()
            labels_data = label_encoder.fit_transform(labels_data)
            class_labels_list = list(label_encoder.classes_)
            class_labels_list.sort()
            return images_data, labels_data, class_labels_list
        else:
            return np.array([]), np.array([]), []

    except Exception as e:
        print(f"Error en load_data: {e}")
        raise e

def create_cnn_model(input_shape, num_classes, learning_rate):
    """
    Crea y compila un modelo CNN más profundo para la clasificación de imágenes a color.

    Args:
        input_shape (tuple): Forma de entrada de las imágenes (ej: (128, 128, 3) para imágenes 128x128 a color).
        num_classes (int): Número de clases a clasificar.
        learning_rate (float): Tasa de aprendizaje para el optimizador.

    Returns:
        tf.keras.Model: Modelo CNN de Keras compilado.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape), # Primera capa convolucional
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'), # Segunda capa convolucional
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'), # Tercera capa convolucional (más profunda)
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'), # Capa densa más grande
        tf.keras.layers.Dropout(0.5), # Dropout para regularización
        tf.keras.layers.Dense(num_classes, activation='softmax') # Capa de salida
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model_kfold(X, y, folds, epochs, learning_rate, class_labels_list):
    """
    Entrena el modelo CNN usando validación cruzada K-Fold.

    Args:
        X (numpy.ndarray): Datos de entrenamiento (imágenes).
        y (numpy.ndarray): Etiquetas de entrenamiento.
        folds (int): Número de pliegues para K-Fold.
        epochs (int): Número de épocas de entrenamiento por pliegue.
        learning_rate (float): Tasa de aprendizaje.
        class_labels_list (list): Lista de etiquetas de clase.

    Returns:
        tuple: (results_data, confusion_matrix_data, best_model_history, best_fold_number, class_labels_list)
               - ... (igual que antes)
    """
    results_data = []
    all_y_true = []
    all_y_pred = []
    best_loss = float('inf')
    best_model_history = None
    best_fold_number = 0

    num_classes = len(np.unique(y))
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    fold_number = 1

    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Crear el modelo CNN con la forma de entrada correcta (imágenes a color 128x128)
        model = create_cnn_model(input_shape=(128, 128, 3), num_classes=num_classes, learning_rate=learning_rate)

        history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0)

        y_pred_probs = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        accuracy = report['accuracy']
        precision = report['macro avg']['precision']
        recall = report['macro avg']['recall']
        f1_score = report['macro avg']['f1-score']

        results_data.append({
            "Pliegue": fold_number,
            "Precisión": accuracy,
            "Precisión_Modelo": precision,
            "Exhaustividad": recall,
            "Puntaje_F1": f1_score
        })

        current_loss = history.history['loss'][-1]
        if current_loss < best_loss:
            best_loss = current_loss
            best_model_history = history
            best_fold_number = fold_number

        fold_number += 1

    confusion_matrix_data = confusion_matrix(all_y_true, all_y_pred)
    return results_data, confusion_matrix_data, best_model_history, best_fold_number, class_labels_list