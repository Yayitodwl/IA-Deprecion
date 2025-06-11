import tkinter as tk
from tkinter import messagebox, filedialog
import pandas as pd
import re
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Modelo y tokenizador para análisis de texto
model_name = 'nlptown/bert-base-multilingual-uncased-sentiment'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Lista global para almacenar comentarios procesados
comentarios_guardados = []

# Función para limpiar el texto (quitar acentos y caracteres especiales)
def limpiar_texto(texto):
    reemplazos = {
        'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
        'Á': 'A', 'É': 'E', 'Í': 'I', 'Ó': 'O', 'Ú': 'U'
    }
    for acento, sin_acento in reemplazos.items():
        texto = texto.replace(acento, sin_acento)
    texto = re.sub(r'[^a-zA-Z0-9\s]', '', texto)  # Eliminar caracteres especiales
    return texto

# Función para detectar emociones en el texto
def detectar_emociones(texto):
    inputs = tokenizer(texto, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    emocion_predicha = torch.argmax(logits, dim=1).item()
    return emocion_predicha

# Función para guardar un comentario en la lista
def guardar_comentario(texto, emocion_predicha):
    emociones = {
        0: "Muy negativa",
        1: "Negativa", 
        2: "Neutral",
        3: "Positiva",
        4: "Muy positiva"
    }
    comentario_limpio = limpiar_texto(texto)
    comentarios_guardados.append({"comentario": comentario_limpio, "emocion": emociones[emocion_predicha]})
    messagebox.showinfo("Guardado", "El comentario ha sido guardado exitosamente.")

# Función para analizar el texto ingresado y guardarlo
def analizar_y_guardar_texto():
    texto = entry_texto.get().strip()
    if texto == "":
        messagebox.showwarning("Advertencia", "Por favor, ingrese texto para analizar.")
    else:
        emocion_predicha = detectar_emociones(texto)
        guardar_comentario(texto, emocion_predicha)
        mostrar_resultado(emocion_predicha)

# Función para descargar todos los comentarios acumulados en un archivo CSV
def descargar_dataset():
    if not comentarios_guardados:
        messagebox.showwarning("Advertencia", "No hay comentarios para descargar.")
    else:
        archivo = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("Archivos CSV", "*.csv")],
            title="Guardar como"
        )
        if archivo:  # Si el usuario selecciona una ubicación
            df = pd.DataFrame(comentarios_guardados)
            df.to_csv(archivo, index=False)
            messagebox.showinfo("Descarga completa", f"El dataset se ha guardado en:\n{archivo}")

# Función para mostrar el resultado de la detección de emociones
def mostrar_resultado(emocion_predicha):
    emociones = {
        0: "depresion muy alta",
        1: "depresión alta",
        2: "Neutral",
        3: "Depresion minima",
        4: "Sin depresión"
    }
    messagebox.showinfo("Análisis de Depresión", f"Nivel de depresión detectado: {emociones[emocion_predicha]}")

# Configuración inicial de la ventana principal
ventana = tk.Tk()
ventana.title("Procesamiento de Texto")

# Título
label_titulo = tk.Label(ventana, text="Escribe uno o varios comentarios en el siguiente recuadro.", font=("Helvetica", 16), fg="black")
label_titulo.pack(pady=10)

# Cuadro de texto para entrada
entry_texto = tk.Entry(ventana, width=50)
entry_texto.pack(pady=10)

# Botones
boton_borrar = tk.Button(ventana, text="Borrar texto", command=lambda: entry_texto.delete(0, 'end'))
boton_borrar.pack(pady=5)

boton_analizar = tk.Button(ventana, text="Analizar y Guardar", command=analizar_y_guardar_texto)
boton_analizar.pack(pady=5)

boton_descargar = tk.Button(ventana, text="Descargar Dataset", command=descargar_dataset)
boton_descargar.pack(pady=10)

# Dimensiones de la ventana
ventana.geometry("500x300")

# Ejecutar el bucle principal de la ventana
ventana.mainloop()
