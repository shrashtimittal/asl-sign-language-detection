import streamlit as st
import torch
import timm
from PIL import Image
from torchvision import transforms

# ----------------- CONFIG -----------------
MODEL_PATH   = r"E:\ASL_Project\models\best_efficientnet_b0.pth"
NUM_CLASSES  = 29
IMG_SIZE     = 224
DEVICE       = "cpu"   # Force CPU
CLASS_NAMES  = sorted([
    'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O',
    'P','Q','R','S','T','U','V','W','X','Y','Z','del','nothing','space'
])
# -------------------------------------------

@st.cache_resource
def load_model():
    model = timm.create_model("efficientnet_b0",
                              pretrained=False,
                              num_classes=NUM_CLASSES)
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

st.title("🖐️ American Sign Language Image Classifier")
st.markdown("Upload an **image of a single ASL sign** to get the predicted letter/word.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    tensor = transform(image).unsqueeze(0)

    # Prediction
    with torch.no_grad():
        outputs = model(tensor)
        pred_idx = outputs.argmax(1).item()
        pred_class = CLASS_NAMES[pred_idx]

    st.success(f"### Prediction: **{pred_class}**")
