import streamlit as st
import torch
import clip
import pandas as pd
import os
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity


@st.cache_resource
def load_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

@st.cache_data
def load_data(results):
    df = pd.read_csv(results, sep='|')
    return df

@st.cache_data
def load_text_features(text_features_weight):
    return torch.load(text_features_weight)



def find_images(query, top, text_features, df):
    # Векторизация текстового запроса
    model, processor = load_model()
    query_input = processor(query, return_tensors="pt")
    query_features = model.get_text_features(**query_input)

    # Поиск самых похожих изображений
    similarity_scores = cosine_similarity(query_features.cpu().detach().numpy(), text_features.cpu().detach().numpy())
    top_indices = similarity_scores.argsort()[0][-top:][::-1]
    top_images = df.loc[top_indices, 'image_name'].tolist()
    top_similarity_scores = similarity_scores[0][top_indices]

    return top_images, top_similarity_scores 

# Основная программа
if __name__ == '__main__':
    st.title("Pictures search")

    images_path = 'flickr30k_images/flickr30k_images/' # в случае архива его надо распоковать, делаю это далее по коду

    # Загрузка модели и данных
    model, processor = load_model()
    df = load_data('flickr30k_images/results.csv')
    text_features = load_text_features('text_features.pt')


    # Объявляем эти переменные заранее, чтобы избежать NameError
    top_images = []
    top_similarity_scores = []

    st.sidebar.header('App Settings')
    num_images = st.sidebar.slider('Number of Search Results', min_value=1, max_value=10)
    user_input = st.sidebar.text_input("Enter text:", "")   
    if st.sidebar.button("Start searching"):
        top_images, top_similarity_scores = find_images(user_input, num_images, text_features, df)


    # Вывод найденных изображений и подписей
    for index, (img_name, score) in enumerate(zip(top_images, top_similarity_scores)):
        st.write(f"Model confidence: <span style='color:red'>{score:.4f}</span>", unsafe_allow_html=True)
        img_path = os.path.join(images_path, img_name)  # уточните путь внутри zip-архива 
        st.image(Image.open(img_path), use_column_width=True)