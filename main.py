# main_app.py
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, Toplevel
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np
import seaborn as sns
import os
from PIL import Image

from model_trainer import load_data, train_model_kfold, create_cnn_model

PRIMARY_COLOR = "#1832a2"
SECONDARY_COLOR = "#aa1717"
BACKGROUND_COLOR = "#d9d9d9"
TEXT_COLOR = "#333333"
ACCENT_COLOR = "#467be3"

class ImageClassificationApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Red Neuronal - Clasificación de Imágenes")
        self.geometry("1100x900")
        self.configure(bg=BACKGROUND_COLOR)

        style = ttk.Style(self)
        style.theme_use('clam')
        style.configure('.', background=BACKGROUND_COLOR, foreground=TEXT_COLOR, font=('Segoe UI', 10))
        style.configure('TButton', background=PRIMARY_COLOR, foreground='white', font=('Segoe UI', 10, 'bold'))
        style.map('TButton', background=[('active', ACCENT_COLOR)])

        self.main_frame = ttk.Frame(self, padding=20)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.config_frame = ttk.LabelFrame(self.main_frame, text="Configuración de Entrenamiento", padding=15)
        self.config_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        self.confusion_data_frame = ttk.LabelFrame(self.main_frame, text="Datos de Matriz de Confusión", padding=15)
        self.confusion_data_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        self.best_model_data_frame = ttk.LabelFrame(self.main_frame, text="Datos del Mejor Modelo", padding=15)
        self.best_model_data_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        self.epochs_var = tk.IntVar(value=100)
        self.folds_var = tk.IntVar(value=5)
        self.learning_rate_var = tk.DoubleVar(value=0.01)
        self.data = None
        self.images_data = []
        self.labels_data = []
        self.class_labels_list = []
        self.results_data = []
        self.confusion_matrix_figure = None
        self.confusion_window = None
        self.best_model_history = None
        self.modified_dataset_window = None
        self.loss_history_window = None
        self.best_fold_number = 0

        self.cm_observations_label_val = tk.StringVar(value="")
        self.cm_fp_label_val = tk.StringVar(value="")
        self.cm_fn_label_val = tk.StringVar(value="")
        self.cm_accuracy_label_val = tk.StringVar(value="")

        self.best_model_training_error_val = tk.StringVar(value="")
        self.best_model_total_error_val = tk.StringVar(value="")

        self.file_label = ttk.Label(self.config_frame, text="Cargar carpeta de imágenes:")
        self.file_label.grid(row=0, column=0, padx=5, pady=5, sticky="e")

        self.status_label = ttk.Label(self.config_frame, text="Carpeta no seleccionada")
        self.status_label.grid(row=0, column=3, padx=5, pady=5, sticky="w")

        self.load_button = ttk.Button(self.config_frame, text="Cargar carpeta de imágenes", command=self.load_image_folder_ui)
        self.load_button.grid(row=0, column=1, padx=10, pady=5)

        self.epochs_label = ttk.Label(self.config_frame, text="Épocas:")
        self.epochs_label.grid(row=1, column=0, padx=5, pady=5, sticky="e")
        self.epochs_entry = ttk.Entry(self.config_frame, textvariable=self.epochs_var, width=10)
        self.epochs_entry.grid(row=1, column=1, padx=5, pady=5)

        self.folds_label = ttk.Label(self.config_frame, text="Modelos (K-Fold):")
        self.folds_label.grid(row=2, column=0, padx=5, pady=5, sticky="e")
        self.folds_entry = ttk.Entry(self.config_frame, textvariable=self.folds_var, width=10)
        self.folds_entry.grid(row=2, column=1, padx=5, pady=5)

        self.learning_rate_label = ttk.Label(self.config_frame, text="Tasa de aprendizaje:")
        self.learning_rate_label.grid(row=3, column=0, padx=5, pady=5, sticky="e")
        self.learning_rate_entry = ttk.Entry(self.config_frame, textvariable=self.learning_rate_var, width=10)
        self.learning_rate_entry.grid(row=3, column=1, padx=5, pady=5)

        self.train_button = ttk.Button(self.config_frame, text="Iniciar Entrenamiento", command=self.train_model_ui)
        self.train_button.grid(row=1, column=2, padx=10, pady=5, rowspan=3)

        self.plot_loss_button = ttk.Button(self.config_frame, text="Mostrar Gráfica de Error", command=self.plot_loss_history_ui)
        self.plot_loss_button.grid(row=1, column=4, padx=10, pady=5)

        self.cm_observations_label = ttk.Label(self.confusion_data_frame, textvariable=self.cm_observations_label_val)
        self.cm_observations_label.pack(pady=2, padx=10, anchor='w')

        self.cm_fp_label = ttk.Label(self.confusion_data_frame, textvariable=self.cm_fp_label_val)
        self.cm_fp_label.pack(pady=2, padx=10, anchor='w')

        self.cm_fn_label = ttk.Label(self.confusion_data_frame, textvariable=self.cm_fn_label_val)
        self.cm_fn_label.pack(pady=2, padx=10, anchor='w')

        self.cm_accuracy_label = ttk.Label(self.confusion_data_frame, textvariable=self.cm_accuracy_label_val)
        self.cm_accuracy_label.pack(pady=2, padx=10, anchor='w')

        self.best_model_training_error_label = ttk.Label(self.best_model_data_frame, textvariable=self.best_model_training_error_val)
        self.best_model_training_error_label.pack(pady=2, padx=10, anchor='w')

        self.best_model_total_error_label = ttk.Label(self.best_model_data_frame, textvariable=self.best_model_total_error_val)
        self.best_model_total_error_label.pack(pady=2, padx=10, anchor='w')

        self.results_frame = ttk.LabelFrame(self.main_frame, text="Resultados del Entrenamiento (K-Fold)", padding=15)
        self.results_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.results_tree = ttk.Treeview(self.results_frame, columns=("Modelo", "Precisión", "Precisión_Modelo", "Exhaustividad", "Puntaje_F1", "Precisión_Entrenamiento"), show="headings")
        columnas_espanol = ["Modelo", "Precisión", "Precisión_Modelo", "Exhaustividad", "Puntaje_F1", "Precisión_Entrenamiento"]
        columnas_ingles = ["Fold", "Accuracy", "Precision", "Recall", "F1-Score", "Training Accuracy"]

        for col_es, col_en in zip(columnas_espanol, columnas_ingles):
            self.results_tree.heading(col_es, text=col_es)
            self.results_tree.column(col_es, width=120, anchor='center')
        self.results_tree.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        self.protocol("WM_DELETE_WINDOW", self.on_closing)


    def verify_image_dimensions(self, folder_path):
        invalid_images = []
        subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]
        for subfolder in subfolders:
            for filename in os.listdir(subfolder):
                if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    continue
                image_path = os.path.join(subfolder, filename)
                try:
                    img = Image.open(image_path)
                    # Ya no verificamos las dimensiones aquí
                except Exception as e:
                    invalid_images.append((filename, os.path.basename(subfolder), str(e))) # Guardamos el error
            return invalid_images

    def load_image_folder_ui(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            try:
                invalid_dimension_images = self.verify_image_dimensions(folder_path) # Ahora solo verifica si se pueden abrir
                if invalid_dimension_images:
                    error_message = "Las siguientes imágenes no se pudieron cargar y serán ignoradas:\n"
                    for filename, folder_name, error_str in invalid_dimension_images: # Ahora también tenemos el error
                        error_message += f"- {filename} en la carpeta '{folder_name}' - Error: {error_str}\n" # Mostramos el error
                    messagebox.showwarning("Advertencia", error_message)

                self.images_data, self.labels_data, self.class_labels_list = load_data(folder_path)
                print("Lista de etiquetas de clase:", self.class_labels_list)

                if len(self.images_data) > 0:
                    self.status_label.config(text=f"Imágenes cargadas: {len(self.images_data)}, Clases: {len(self.class_labels_list)}", foreground="green")
                else:
                    self.status_label.config(text="No se encontraron imágenes válidas", foreground="red")
                    self.images_data = []
                    self.labels_data = []
                    self.class_labels_list = []

            except Exception as e:
                messagebox.showerror("Error", f"Error al cargar las imágenes: {e}")
                self.status_label.config(text="Error al cargar imágenes", foreground="red")
                self.images_data = []
                self.labels_data = []
                self.class_labels_list = []
        else:
            self.status_label.config(text="Carpeta no seleccionada", foreground="red")
            self.images_data = []
            self.labels_data = []
            self.class_labels_list = []

    def train_model_ui(self):
        if not (hasattr(self.images_data, 'size') and self.images_data.size > 0) or not (hasattr(self.labels_data, 'size') and self.labels_data.size > 0):
            messagebox.showerror("Error", "No se han cargado imágenes. Por favor, carga una carpeta de imágenes.")
            return

        self.results_data = []
        self.results_tree.delete(*self.results_tree.get_children())

        epochs = self.epochs_var.get()
        folds = self.folds_var.get()
        learning_rate = self.learning_rate_var.get()

        try:
            results_data, confusion_matrix_data, best_model_history, best_fold_number, class_labels_list = train_model_kfold(
                self.images_data, self.labels_data, folds, epochs, learning_rate, self.class_labels_list # Pass self.class_labels_list here!
            )
            self.results_data = results_data
            self.confusion_matrix_data = confusion_matrix_data
            self.best_model_history = best_model_history
            self.best_fold_number = best_fold_number
            self.class_labels_list = class_labels_list

            for result in self.results_data:
                self.results_tree.insert("", "end", values=(
                    result["Pliegue"],
                    f"{result['Precisión']:.4f}",
                    f"{result['Precisión_Modelo']:.4f}",
                    f"{result['Exhaustividad']:.4f}",
                    f"{result['Puntaje_F1']:.4f}",
                    "-"
                ))

            total_observations = np.sum(self.confusion_matrix_data)
            false_positives = np.sum(self.confusion_matrix_data, axis=0) - np.diag(self.confusion_matrix_data)
            false_negatives = np.sum(self.confusion_matrix_data, axis=1) - np.diag(self.confusion_matrix_data)
            total_false_positives = np.sum(false_positives)
            total_false_negatives = np.sum(false_negatives)
            training_accuracy = np.trace(self.confusion_matrix_data) / total_observations if total_observations else 0
            training_accuracy_percent = f"{training_accuracy * 100:.2f}%"

            self.cm_observations_label_val.set(f"Cantidad de Observaciones: {total_observations}")
            self.cm_fp_label_val.set(f"Falsos Positivos (Total): {total_false_positives}")
            self.cm_fn_label_val.set(f"Falsos Negativos (Total): {total_false_negatives}")
            self.cm_accuracy_label_val.set(f"Precisión Matriz Confusión: {training_accuracy_percent}")

            last_item = self.results_tree.get_children()[-1]
            self.results_tree.item(last_item, values=self.results_tree.item(last_item, 'values')[0:5] + (training_accuracy_percent,))

            self.show_confusion_matrix()

            best_model_training_loss = self.best_model_history.history['loss'][-1] if self.best_model_history else "N/A"
            best_model_total_accuracy_percent = training_accuracy_percent
            best_fold_num_display = self.best_fold_number

            self.best_model_training_error_val.set(f"Error de Entrenamiento (Mejor Modelo - Pliegue {best_fold_num_display}): {best_model_training_loss:.4f}") # Mostrar el pliegue del mejor modelo
            self.best_model_total_error_val.set(f"Error Total (Precisión Matriz Confusión): {training_accuracy_percent}")


            messagebox.showinfo("Éxito", "Entrenamiento completado. La matriz de confusión detallada se muestra en otra ventana y los datos resumidos en la ventana principal.")

        except Exception as e:
            messagebox.showerror("Error", f"Error durante el entrenamiento: {e}")
            print(f"Error en train_model_ui: {e}")


    def plot_loss_history_ui(self):
        if self.best_model_history is None:
            messagebox.showinfo("Información", "No hay historial de pérdida disponible. Entrena el modelo primero.")
            return

        if self.loss_history_window:
            self.loss_history_window.destroy()

        self.loss_history_window = Toplevel(self)
        self.loss_history_window.title("Evolución del Error (Pérdida) del Mejor Modelo")

        loss_figure = plt.figure(figsize=(8, 6))
        ax_loss = loss_figure.add_subplot(111)
        ax_loss.plot(self.best_model_history.history['loss'])
        ax_loss.set_title('Evolución del Error (Pérdida) durante el Entrenamiento')
        ax_loss.set_xlabel('Épocas')
        ax_loss.set_ylabel('Pérdida')
        ax_loss.grid(True)

        canvas_loss = FigureCanvasTkAgg(loss_figure, master=self.loss_history_window)
        canvas_loss_widget = canvas_loss.get_tk_widget()
        canvas_loss_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
        canvas_loss.draw()


    def show_confusion_matrix(self):
        if self.confusion_window:
            self.confusion_window.destroy()
        self.confusion_window = Toplevel(self)
        self.confusion_window.title("Matriz de Confusión")

        self.confusion_matrix_figure = plt.figure(figsize=(8, 7))
        ax = self.confusion_matrix_figure.add_subplot(111)
        class_labels = self.class_labels_list
        print("Clases: ", class_labels)
        cm_display = ConfusionMatrixDisplay(confusion_matrix=self.confusion_matrix_data, display_labels=class_labels)
        cm_display.plot(ax=ax, cmap=plt.cm.Blues)
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Valor Real") 
        ax.set_title("Matriz de Confusión")

        canvas_cm = FigureCanvasTkAgg(self.confusion_matrix_figure, master=self.confusion_window)
        canvas_cm_widget = canvas_cm.get_tk_widget()
        canvas_cm_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
        canvas_cm.draw()


    def on_closing(self):
        if messagebox.askokcancel("Salir", "¿Seguro que quieres salir?"):
            self.destroy()


if __name__ == "__main__":
    app = ImageClassificationApp()
    app.mainloop()