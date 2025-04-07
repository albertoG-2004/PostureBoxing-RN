import tensorflow as tf
import numpy as np
import os
from PIL import Image
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight

def load_data(folder_path):
    """
    Carga imágenes a color desde carpetas de clases directamente
    en la carpeta principal. Redimensiona las imágenes a un tamaño mayor (128x128).

    Args:
        folder_path (str): Ruta a la carpeta principal del dataset (ej: 'dataset').

    Returns:
        tuple: (images_data, labels_data, class_labels_list)
               - images_data (numpy.ndarray): Array con los datos de las imágenes redimensionadas y normalizadas.
               - labels_data (numpy.ndarray): Array con las etiquetas numéricas correspondientes a las clases.
               - class_labels_list (list): Lista de nombres de las clases (etiquetas originales).
    """
    images_data = []
    labels_data = []
    class_labels_list = []
    target_size = (128, 128)  # Mantenemos la resolución original para compatibilidad

    try:
        class_folders = [f.path for f in os.scandir(folder_path) if f.is_dir()]
        if not class_folders:
            raise ValueError("La carpeta debe contener subcarpetas con las clases de boxeo (Cruzado, Gancho derecha, etc.).")

        class_labels_set = set()
        class_counts = {}  # Para monitorear el balance de clases

        for class_folder in class_folders:
            class_name = os.path.basename(class_folder)
            class_labels_set.add(class_name)
            class_counts[class_name] = 0
            
            for filename in os.listdir(class_folder):
                if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    continue
                    
                image_path = os.path.join(class_folder, filename)
                try:
                    img = Image.open(image_path)
                    # Convertir imágenes a RGB para asegurar 3 canales
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                        
                    img = img.resize(target_size, Image.LANCZOS)
                    img_array = np.array(img) / 255.0  # Normalizamos los valores de color (0-1)
                    
                    # Verificar que la imagen tenga la forma correcta
                    if img_array.shape == (target_size[0], target_size[1], 3):
                        images_data.append(img_array)
                        labels_data.append(class_name)
                        class_counts[class_name] += 1
                    else:
                        print(f"Imagen {filename} con forma incorrecta: {img_array.shape}")
                        
                except Exception as e:
                    print(f"Error al cargar imagen {filename}: {e}")

        # Mostrar distribución de clases
        print("Distribución de clases:")
        for class_name, count in class_counts.items():
            print(f"  {class_name}: {count} imágenes")

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
    Crea y compila un modelo CNN mejorado para la clasificación de imágenes a color.

    Args:
        input_shape (tuple): Forma de entrada de las imágenes (ej: (128, 128, 3) para imágenes 128x128 a color).
        num_classes (int): Número de clases a clasificar.
        learning_rate (float): Tasa de aprendizaje para el optimizador.

    Returns:
        tf.keras.Model: Modelo CNN de Keras compilado.
    """
    model = tf.keras.Sequential([
        # Primer bloque convolucional
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        # Segundo bloque convolucional
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        # Tercer bloque convolucional
        tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        # Capas densas
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def create_data_augmentation():
    """
    Crea un generador de aumento de datos para mejorar la generalización.
    
    Returns:
        tf.keras.Sequential: Capa de aumento de datos.
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1),
        tf.keras.layers.RandomBrightness(0.1)
    ])

def train_model_kfold(X, y, folds, epochs, learning_rate, class_labels_list):
    """
    Entrena el modelo CNN usando validación cruzada K-Fold con mejoras.

    Args:
        X (numpy.ndarray): Datos de entrenamiento (imágenes).
        y (numpy.ndarray): Etiquetas de entrenamiento.
        folds (int): Número de pliegues para K-Fold.
        epochs (int): Número de épocas de entrenamiento por pliegue.
        learning_rate (float): Tasa de aprendizaje.
        class_labels_list (list): Lista de etiquetas de clase.

    Returns:
        tuple: (results_data, confusion_matrix_data, best_model_history, best_fold_number, class_labels_list)
               - results_data: Lista de diccionarios con métricas por pliegue
               - confusion_matrix_data: Matriz de confusión global
               - best_model_history: Historial del mejor modelo
               - best_fold_number: Número del mejor pliegue
               - class_labels_list: Lista de etiquetas de clase
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

    # Crear capa de aumento de datos
    data_augmentation = create_data_augmentation()

    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Calcular pesos de clase para manejar desbalance
        class_weights = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}

        # Crear el modelo CNN con la forma de entrada correcta
        input_shape = X_train[0].shape
        model = create_cnn_model(input_shape=input_shape, num_classes=num_classes, learning_rate=learning_rate)

        # Callbacks para mejorar el entrenamiento
        callbacks = [
            # Reducir learning rate cuando se estanca
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]

        # Preparar generador con aumento de datos
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_dataset = train_dataset.shuffle(buffer_size=len(X_train))
        train_dataset = train_dataset.batch(32)
        train_dataset = train_dataset.map(
            lambda x, y: (data_augmentation(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

        # Entrenar el modelo
        history = model.fit(
            train_dataset,
            epochs=epochs,
            verbose=0,
            callbacks=callbacks,
            class_weight=class_weights_dict
        )

        # Evaluar el modelo
        y_pred_probs = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

        # Calcular métricas
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        accuracy = report['accuracy']
        precision = report['macro avg']['precision']
        recall = report['macro avg']['recall']
        f1_score = report['macro avg']['f1-score']

        # Guardar resultados
        results_data.append({
            "Pliegue": fold_number,
            "Precisión": accuracy,
            "Precisión_Modelo": precision,
            "Exhaustividad": recall,
            "Puntaje_F1": f1_score
        })

        # Guardar el mejor modelo
        current_loss = history.history['loss'][-1]
        if current_loss < best_loss:
            best_loss = current_loss
            best_model_history = history
            best_fold_number = fold_number

        fold_number += 1

    # Calcular matriz de confusión global
    confusion_matrix_data = confusion_matrix(all_y_true, all_y_pred)
    
    return results_data, confusion_matrix_data, best_model_history, best_fold_number, class_labels_list