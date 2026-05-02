import streamlit as st
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from torchvision import models, transforms
from PIL import Image

# ========================
# CONFIG
# ========================
DEVICE = torch.device("cpu")

st.set_page_config(
    page_title="Anti-Hoax Detector",
    page_icon="📰",
    layout="wide"
)

# ========================
# MODEL
# ========================
class MultiModal(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained("indobenchmark/indobert-base-p1")
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        for p in self.resnet.parameters():
            p.requires_grad = False

        self.resnet.fc = nn.Identity()

        self.fc = nn.Sequential(
            nn.Linear(768 + 2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )

    def forward(self, ids, mask, img):
        t = self.bert(ids, attention_mask=mask).last_hidden_state[:,0,:]
        i = self.resnet(img)
        return self.fc(torch.cat((t,i),1))

@st.cache_resource
def load_model():
    model = MultiModal()
    model.load_state_dict(torch.load("best.pth", map_location=DEVICE))
    model.eval()
    return model

model = load_model()

# ========================
# TOKENIZER
# ========================
tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")

def tokenize(text):
    return tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors='pt'
    )

# ========================
# IMAGE TRANSFORM
# ========================
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ========================
# PREDICT
# ========================
def predict(text, image):
    enc = tokenize(text)

    ids = enc['input_ids'].to(DEVICE)
    mask = enc['attention_mask'].to(DEVICE)

    img = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(ids, mask, img)
        pred = torch.argmax(output,1).item()

    return pred

# ========================
# UI
# ========================
st.title("📰 Anti-Hoax News Detector")

col1, col2 = st.columns([2,1])

with col1:
    text_input = st.text_area("Masukkan teks berita")

with col2:
    image_input = st.file_uploader("Upload gambar", type=["jpg","png","jpeg"])

if st.button("🔍 Analisis"):
    if text_input and image_input:
        image = Image.open(image_input).convert("RGB")
        result = predict(text_input, image)

        st.image(image)

        if result == 1:
            st.error("🚨 HOAX")
        else:
            st.success("✅ VALID")

    else:
        st.warning("Isi semua input!")