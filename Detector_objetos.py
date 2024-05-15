#bibliotecas necesarias
import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes

# Definir los pesos predeterminados y las categorías
weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT #modelo pre entrando pára deteccion de objetos
categories = weights.meta["categories"] ## Lista de categorías de objetos

# Preprocesar la imagen para que esté en el rango [0,1]
img_preprocess = weights.transforms() ## Escala los valores de 0-255 a 0-1

# Cargar el modelo de detección de objetos con Fast R-CNN y ResNet-50 FPN pre-entrenado
@st.cache_resource
def load_model():
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.5)
    model.eval(); ## Establecer el modelo en modo de evaluación
    return model

model = load_model()

# Función para realizar predicciones sobre la imagen cargada
def make_prediction(img): 
    img_processed = img_preprocess(img) ## Preprocesar la imagen
    prediction = model(img_processed.unsqueeze(0)) # Realizar la predicción
    prediction = prediction[0]                       ## Diccionario con claves "boxes", "labels", "scores".
    prediction["labels"] = [categories[label] for label in prediction["labels"]] ## Convertir los índices de etiquetas a nombres de categorías
    return prediction

# Función para crear una imagen con las cajas delimitadoras alrededor de los objetos detectados
def create_image_with_boxes(img, prediction): 
    img_tensor = torch.tensor(img) ## Convertir la imagen a tensor
    img_with_bboxes = draw_bounding_boxes(img_tensor, boxes=prediction["boxes"], labels=prediction["labels"],
                                          colors=["red" if label=="person" else "green" for label in prediction["labels"]] , width=2)
    img_with_boxes_np = img_with_bboxes.detach().numpy().transpose(1,2,0) ### Cambiar el formato de la imagen de (3,W,H) a (W,H,3), de canal primero a canal último.
    return img_with_boxes_np

## Panel de control
st.title("Detector de objetos")
upload = st.file_uploader(label="Sube tu imagen", type=["png", "jpg", "jpeg"])

if upload:
    img = Image.open(upload) ## Abrir la imagen cargada

    prediction = make_prediction(img) ## Realizar la predicción
    img_with_bbox = create_image_with_boxes(np.array(img).transpose(2,0,1), prediction) ## Cambiar el formato de la imagen

    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111)
    plt.imshow(img_with_bbox)

    # Ajustar el tamaño de fuente para el texto dentro de las cajas delimitadoras
    fontsize = 14

    for box, label in zip(prediction["boxes"], prediction["labels"]):
        x, y, w, h = box
        ax.text(x, y, label, color='white', fontsize=fontsize, bbox=dict(facecolor='red', alpha=0.5))

    plt.xticks([],[])
    plt.yticks([],[])
    ax.spines[["top", "bottom", "right", "left"]].set_visible(False)

    st.pyplot(fig, use_container_width=True) ## Mostrar la imagen en el panel

    del prediction["boxes"] ## Eliminar las cajas delimitadoras de la predicción

    #st.header("Probabilidades predichas") 
    st.write(prediction["labels"]) ## Mostrar las etiquetas de las predicciones sin los puntajes (scores)