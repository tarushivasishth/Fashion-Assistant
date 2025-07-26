import streamlit as st
import numpy as np
import pandas as pd
import os
import joblib
import gzip
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
from mapping import gender_map, base_colour_map, season_map, complementary_map
import gdown
import zipfile


st.set_page_config(page_title="AI Stylist", layout="wide")

# === CONFIG ===
IMAGE_FOLDER = "images"
ZIP_PATH = "images.zip"

# === DOWNLOAD AND UNZIP IMAGES IF NOT ALREADY AVAILABLE ===
if not os.path.exists(IMAGE_FOLDER):
    st.info("📦 Downloading image assets...")
    gdown.download(f"https://drive.google.com/uc?id=100QthkMdfv6rQtHRFKE1ME9y1ux2Q23h", ZIP_PATH, quiet=False)

    st.info("🗃️ Unzipping files...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall('.')

    st.success("✅ Images ready!")

df = pd.read_csv("datasets/df_balanced.csv")

resnet = ResNet50(weights="imagenet", include_top=False, pooling="avg")
feature_model = Model(inputs=resnet.input, outputs=resnet.output)

def extract_features(img_path):
    try:
        img = load_img(img_path, target_size=(224,224))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0) # convert to 2D
        x = preprocess_input(x)
        feat = feature_model.predict(x,verbose=0)
        return feat.flatten() # convert back to 1D
    except:
        return np.zeros(2048)
    
def load_models_encoders(model_dir):
    encoders = {}
    models = {}

    for file in os.listdir(model_dir):
        path = os.path.join(model_dir, file)

        if file.endswith("_model.pkl.gz"):
            key = file.replace("_model.pkl.gz", "")
            with gzip.open(path, 'rb') as f:
                models[key] = joblib.load(f)

        elif file.endswith("_encoder.pkl.gz"):
            key = file.replace("_encoder.pkl.gz", "")
            with gzip.open(path, 'rb') as f:
                encoders[key] = joblib.load(f)

    return encoders, models
# def load_models_encoders(model_dir):
#     encoders = {}
#     models = {}

#     for file in os.listdir(model_dir):
#         if file.endswith("_model.pkl"):
#             key = file.replace("_model.pkl", "")
#             models[key] = joblib.load(os.path.join(model_dir, file))
#         elif file.endswith("_encoder.pkl"):
#             key = file.replace("_encoder.pkl", "")
#             encoders[key] = joblib.load(os.path.join(model_dir, file))
    
#     return encoders, models

# Features to use for similarity
similarity_cols = ['baseColour', 'gender', 'season', 'usage']

# Fit encoder once on your DataFrame
encoder = OneHotEncoder(handle_unknown='ignore')
encoder.fit(df[similarity_cols])

def get_complementary_base_colours(base_colour):
    return base_colour_map.get(base_colour, [base_colour])

def get_matching_gender(gender):
    return gender_map.get(gender, [gender])

def get_matching_season(season):
    return season_map.get(season, [season])


def recommend_outfit_items(pred_labels, df, image_dir='images', top_k=5):
    subcat      = pred_labels['subCategory']
    article     = pred_labels['articleType']
    base_colour = pred_labels['baseColour']
    gender      = pred_labels['gender']
    season      = pred_labels['season']
    usage       = pred_labels['usage']

    # Matching gender and season logic
    valid_genders = get_matching_gender(gender)
    valid_seasons = get_matching_season(season)
    valid_colours = get_complementary_base_colours(base_colour)

    # Filter based on rules
    filtered_df = df[
        (df['subCategory'] != subcat) &
        (df['gender'].isin(valid_genders)) &
        (df['season'].isin(valid_seasons)) &
        (df['baseColour'].isin(valid_colours)) &
        (df['usage'] == usage)
    ].copy()

    if filtered_df.empty:
        st.warning("No complementary items found.")
        return

    # One-hot encode input and filtered items
    similarity_cols = ['baseColour', 'gender', 'season', 'usage']

    input_row = pd.DataFrame([{
        'baseColour': base_colour,
        'gender': gender,
        'season': season,
        'usage': usage
    }])

    X_input = encoder.transform(input_row[similarity_cols]).toarray()
    X_cat   = encoder.transform(filtered_df[similarity_cols]).toarray()

    # Compute similarity
    sims = cosine_similarity(X_input, X_cat)[0]
    filtered_df['similarity'] = sims

    # Filter valid article type recommendations
    valid_articles = complementary_map.get(article)
    filtered_df = filtered_df[filtered_df['articleType'].isin(valid_articles)]

    if filtered_df.empty:
        st.warning("No matching article types found.")
        return

    # Pick top 1 per subcategory
    top_items = (
        filtered_df
        .sort_values('similarity', ascending=False)
        .groupby('subCategory', as_index=False)
        .first()
        .sort_values('similarity', ascending=False)
        .head(top_k)
    )

    # 🎯 Streamlit display
    st.subheader("🎯 Recommended Outfit Items")

    for _, row in top_items.iterrows():
        col1, col2 = st.columns([1, 2])

        img_path = os.path.join(image_dir, row['filename'])
        if os.path.exists(img_path):
            with col1:
                st.image(Image.open(img_path), width=180)

        with col2:
            st.markdown(f"**🧢 Product:** {row['productDisplayName']}")
            st.markdown(f"**👕 SubCategory:** {row['subCategory']}")
            st.markdown(f"**📄 Article Type:** {row['articleType']}")
            st.markdown(f"**🎨 Colour:** {row['baseColour']}")
            st.markdown(f"**🧍 Gender:** {row['gender']}")
            st.markdown(f"**🌦 Season:** {row['season']}")
            st.markdown(f"**🎯 Usage:** {row['usage']}")
            st.markdown("---")
