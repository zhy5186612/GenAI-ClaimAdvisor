import streamlit as st
from utils import (
    imageb64_to_clip_embedding, 
    text_to_clip_embedding,
    load_images,
)

import base64
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO
from sklearn.metrics.pairwise import cosine_similarity

IMG_HTML = '<img src="data:image/png;base64, {b64}"/>'

def calculate_image_similarities(images: pd.DataFrame, embedding: list) -> pd.DataFrame:
    images["sims"] = images.apply(
        lambda r: cosine_similarity([r["embedding"], embedding])[0][1], axis=1
    )
    images.sort_values("sims", ascending=False, inplace=True)
    images.reset_index(drop=True, inplace=True)
    return images

def apply_maxdiff_research_filter(filter: dict, images: pd.DataFrame) -> pd.DataFrame:
    if not filter:
        return images
    
    for research, checked in filter.items():
        if not checked:
            images = images.loc[images.research != research]
    images = images if len(images) else load_images()
    
    return images

def show_maxdiff_research_filter(images: pd.DataFrame) -> dict:
    filter = {}
    if st.sidebar.toggle("Research"):
        for r in images.research.unique():
            filter[r] = st.sidebar.checkbox(r, False)
    return filter

def show():
    images = load_images()
    if 'query' not in st.session_state:
        st.session_state.query = ""

    with st.form(key='query_form'):
        query = st.text_input("Type some keywords for your claim", value=st.session_state.query)
        st.session_state.query = query
        st.form_submit_button(label='Search')

    img_file_buffer = st.file_uploader('Upload a image')
    weight = st.sidebar.slider("Weight of Image & Query", 
                       min_value=0.0, max_value=1.0, value=0.5)
    num_results = st.sidebar.slider("Number of results to display", 1, 50, 5)
    research_filter = show_maxdiff_research_filter(images)
    images = apply_maxdiff_research_filter(research_filter, images)
    img_embedding = np.array([0])
    text_embedding = np.array([0])
    
    if img_file_buffer:
        img = Image.open(img_file_buffer)
        img_bytes = BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes = img_bytes.getvalue()
        img_b64 = base64.b64encode(img_bytes).decode()
        img_embedding = imageb64_to_clip_embedding(img_b64)
        st.markdown("### Uploaded Image:")
        st.markdown(IMG_HTML.format(b64=img_b64), unsafe_allow_html=True)
        
    if query:
        text_embedding = text_to_clip_embedding(query)

    if img_file_buffer and query:
        # Adjust embedding by weight if both text and image are given
        total_embedding = (1 - weight) * img_embedding + weight *text_embedding
    else:
        # Otherwise, use one of them as embedding, x + 0 = x
        total_embedding = img_embedding + text_embedding

    if img_file_buffer or query:
        images = calculate_image_similarities(images, total_embedding)
        images["Image"] = images.b64.apply(lambda s: IMG_HTML.format(b64=s))
        images = images[["Image", "sims", "maxdiff", "research"]]
        images.columns = ["Image", "Similarity", "MaxDiff Score", "Research"]
        images = images.iloc[:num_results]
        images.sort_values(["MaxDiff Score"], inplace=True, ascending=False)
        images.reset_index(drop=True, inplace=True)
        images.index += 1
        st.session_state.similar_images = images

    if 'similar_images' in st.session_state:
        st.markdown("### Similar Images:")
        st.markdown(st.session_state.similar_images.to_html(escape=False, justify="center"), unsafe_allow_html=True)