import streamlit as st
import os
import numpy as np
import pandas as pd
from PIL import Image
import joblib
from utils import recommend_outfit_items, load_models_encoders, extract_features

# load dataframe(preprocessed and balanced one)
df = pd.read_csv("datasets/df_balanced.csv")

# load label encoders and models for each target
encoders, models = load_models_encoders("saved_models/")

# image upload
uploaded_file = st.file_uploader("Upload a clothing item image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Item", width=300)

    # extract features from uploaded image
    with st.spinner("Extracting features..."):
        uploaded_feat = extract_features(uploaded_file)

    # predict each label
    predictions = {}
    for label, model in models.items():
        pred_label = model.predict([uploaded_feat.flatten()])[0]
        decoded_class = encoders[label].inverse_transform([pred_label])[0]
        predictions[label] = decoded_class
    
    # display predicted tags
    st.markdown("Predicted Attributes")
    st.write(predictions)

    # recommend matching items
    st.markdown("Recommended Outfit Items")
    with st.spinner("Finding Recommendations..."):
        recommend_outfit_items(predictions, df, image_dir='images/')