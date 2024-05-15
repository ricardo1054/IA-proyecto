import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes

# Definir los pesos predeterminados y las categorías
weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
categories = weights.meta["categories"]

# Preprocesar la imagen para que esté en el rango [0,1]
img_preprocess = weights.transforms()

# Cargar el modelo de detección de objetos con Fast R-CNN y ResNet-50 FPN pre-entrenado
@st.cache_resource
def load_model():
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.5)
    model.eval()
    return model

model = load_model()

# Función para realizar predicciones sobre la imagen cargada
def make_prediction(img):
    img = img.convert("RGB")  # Asegurar que la imagen esté en modo RGB
    img_tensor = img_preprocess(img)
    with torch.no_grad():
        prediction = model([img_tensor])
    prediction = prediction[0]
    prediction["labels"] = [categories[label] for label in prediction["labels"]]
    return prediction

# Función para crear una imagen con las cajas delimitadoras alrededor de los objetos detectados
def create_image_with_boxes(img, prediction):
    img_tensor = torch.tensor(np.array(img)).permute(2, 0, 1)  # Convertir la imagen a tensor y cambiar el formato
    img_with_bboxes = draw_bounding_boxes(img_tensor, boxes=prediction["boxes"], labels=prediction["labels"],
                                          colors=["red" if label == "person" else "green" for label in prediction["labels"]], width=2)
    img_with_boxes_np = img_with_bboxes.permute(1, 2, 0).numpy()  # Cambiar el formato de la imagen de (3, W, H) a (W, H, 3)
    return img_with_boxes_np

# Panel de control
st.title("Detector de objetos con IA")
st.markdown("---")

st.sidebar.title("Configuración")
upload = st.sidebar.file_uploader(label="Sube tu imagen", type=["png", "jpg", "jpeg"])

if upload:
    img = Image.open(upload)

    prediction = make_prediction(img)
    img_with_bbox = create_image_with_boxes(img, prediction)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img_with_bbox)

    # Ajustar el tamaño de fuente para el texto dentro de las cajas delimitadoras
    fontsize = 14

    for box, label in zip(prediction["boxes"], prediction["labels"]):
        x, y, w, h = box
        ax.text(x, y, label, color='white', fontsize=fontsize, bbox=dict(facecolor='red', alpha=0.5))

    ax.axis('off')

    st.pyplot(fig, use_container_width=True)

    del prediction["boxes"]

    st.markdown("---")
    st.header("Resultados")
    st.write(prediction["labels"])
