import streamlit as st
import torch
import clip
import pandas as pd
import os
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import io
import zipfile
import numpy as np

device = 'cpu'
zip_path = 'flickr.zip'

@st.cache_resource
def load_model_photo():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, preprocess

@st.cache_resource
def load_model_text():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

@st.cache_data
def load_data(results):
    df = pd.read_csv(results, sep='|')
    return df

@st.cache_data
def load_features(features_weight):
    return torch.load(features_weight, map_location=torch.device('cpu'))

def unpack_images(zip_path):
    if not os.path.exists(images_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall('.')

def find_images(query, top, text_features, df):
    # Векторизация текстового запроса
    model_text, processor = load_model_text()
    query_input = processor(query, return_tensors="pt")
    query_features = model_text.get_text_features(**query_input)
    # print(np.array(text_features).reshape(-1, 1).shape, query_features.cpu().detach().numpy().shape)

    # Поиск самых похожих изображений
    similarity_scores = cosine_similarity(query_features.cpu().detach().numpy(), text_features.cpu().detach().numpy())
    top_indices = similarity_scores.argsort()[0][-top:][::-1]
    top_images = df.loc[top_indices, 'image_name'].tolist()
    top_similarity_scores = similarity_scores[0][top_indices]

    return top_images, top_similarity_scores

def find_images_by_photo(file_name, num_images, images_features, images_path):
    model_photo, preprocess = load_model_photo()

    image = Image.open(file_name)
    image_input = preprocess(
        images=[image],  # Здесь передаем изображение как список
        return_tensors="pt")
    
    with torch.no_grad():
        image_features = model_photo.get_image_features(**image_input)

    sim = cosine_similarity(image_features.cpu().detach().numpy(), images_features.cpu().detach().numpy())
    top_indices = sim.argsort()[0][-num_images:][::-1]
    top_similarity_scores = sim[0][top_indices]

    image_index_to_filename = {}
    for idx, filename in enumerate(os.listdir(images_path)):
        image_index_to_filename[idx] = filename
    top_image_paths = [image_index_to_filename[idx] for idx in top_indices]

    return top_image_paths, top_similarity_scores

genre = st.sidebar.radio(
    "**How you would find the images?**",
    ["Text", "Photo :movie_camera:"])

if genre == 'Text':
    st.title("Finally find that same picture!")

    images_path = 'flickr30k_images/flickr30k_images' # в случае архива его надо распоковать, делаю это далее по коду

    # Загрузка модели и данных
    model_text, processor = load_model_text()
    df = load_data('results.csv')
    text_features = load_features('text_features.pt')

    top_images = []
    top_similarity_scores = []

    st.sidebar.write('**Settings**')
    num_images = st.sidebar.slider('Number of Search Results', min_value=1, max_value=10)
    user_input = st.sidebar.text_input("Enter text:", "")

    unpack_images(zip_path)

    if st.sidebar.button("Search!"):
        top_images, top_similarity_scores = find_images(user_input, num_images, text_features, df)

    for index, (img_name, score) in enumerate(zip(top_images, top_similarity_scores)):
        st.write(f"Model confidence: <span style='color:red'>{score:.4f}</span>", unsafe_allow_html=True)
        img_path = os.path.join(images_path, img_name)  # уточните путь внутри zip-архива 
        st.image(Image.open(img_path), use_column_width=True)
else:
    st.title("Finally find that same picture!")

    images_path = 'flickr30k_images/flickr30k_images'
    images_features = load_features('image_features.pt')
    model_photo, preprocess = load_model_photo()

    top_images = []
    top_similarity_scores = []

    st.sidebar.header('App Settings')

    num_images = st.sidebar.slider('Number of Search Results', min_value=1, max_value=10)
    image_file = st.sidebar.file_uploader("Upload the image", type=["jpg", "png", "jpeg"])

    if image_file is not None:
        # Создаем временный буфер в памяти
        file_name = io.BytesIO()
        file_name.write(image_file.read())

    unpack_images(zip_path)

    if st.sidebar.button("Search!"):
        top_images, top_similarity_scores = find_images_by_photo(file_name, num_images, images_features, images_path)

    for index, (img_name, score) in enumerate(zip(top_images, top_similarity_scores)):
        st.write(f"Model confidence: <span style='color:red'>{score:.4f}</span>", unsafe_allow_html=True)
        img_path = os.path.join(images_path, img_name)
        st.image(Image.open(img_path), use_column_width=True)