import streamlit as st
import torch
import timm
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

# --------------------
# Page Setup
# --------------------
st.set_page_config(page_title="Mosquito Larvae Classifier", layout="centered")
st.markdown("<h1 style='text-align: center;'>🦟 Mosquito Larvae Classification</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Upload an image and choose a model to identify Aedes, Anopheles, or Culex</p>", unsafe_allow_html=True)

# --------------------
# Globals
# --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ['Aedes', 'Anopheles', 'Culex']

# --------------------
# Sidebar
# --------------------
with st.sidebar:
    st.header("📋 Options")
    model_choice = st.selectbox("Choose Model", ["MobileViT", "ResNet50", "YOLOv8m"])
    st.markdown("---")
    st.markdown("🔍 **Model Info:**")
    st.markdown({
        "MobileViT": "- Combines CNN & Vision Transformer\n- Lightweight and accurate",
        "ResNet50": "- Deep residual CNN\n- Strong baseline for image tasks",
        "YOLOv8m": "- Real-time object detector\n- Returns bounding boxes + class"
    }[model_choice])
    st.markdown("---")
    st.caption("Developed by Md Bodrul Islam")

# --------------------
# Image Upload
# --------------------
uploaded_file = st.file_uploader("📤 Upload a mosquito larva image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    img_tensor = transform(image).unsqueeze(0).to(device)

    if model_choice == "MobileViT":
        model = timm.create_model("mobilevit_s", pretrained=False, num_classes=3)
        model.load_state_dict(torch.load("models/MobileViT.pth", map_location=device))
        model.to(device).eval()

        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)
            result = class_names[pred.item()]
            st.success(f"✅ Prediction: **{result}** ({conf.item()*100:.2f}% confidence)")

    elif model_choice == "ResNet50":
        model = models.resnet50(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, 3)
        model.load_state_dict(torch.load("models/ResNet50.pth", map_location=device))
        model.to(device).eval()

        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)
            result = class_names[pred.item()]
            st.success(f"✅ Prediction: **{result}** ({conf.item()*100:.2f}% confidence)")

    elif model_choice == "YOLOv8m":
        model = YOLO("models/yolov8m.pt")
        results = model(image)
        names = results[0].names

        if results[0].boxes:
            b = results[0].boxes[0]
            cls = int(b.cls.item())
            conf = b.conf.item()
            result = names[cls]

            # Draw bounding box
            img_draw = image.copy()
            draw = ImageDraw.Draw(img_draw)
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1, y1 - 10), f"{result} ({conf*100:.2f}%)", fill="red")

            st.image(img_draw, caption="📌 YOLOv8 Prediction", use_container_width=True)
            st.success(f"✅ Detected: **{result}** ({conf*100:.2f}% confidence)")
        else:
            st.warning("⚠️ No mosquito larvae detected.")

# --------------------
# Footer
# --------------------
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>Made with ❤️ using Streamlit, PyTorch, and YOLOv8</p>",
    unsafe_allow_html=True
)
