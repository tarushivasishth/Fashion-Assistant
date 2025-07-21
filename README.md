FASHION ASSISTANT - OUTFIT RECOMMENDATION SYSTEM
================================================

![App Screenshot][app_ss.png] 

Fashion Assistant is an AI-powered outfit recommendation app that helps users find the best clothing combinations based on input images. It uses a pre-trained ResNet50 model to extract image features and suggest matching items. The app is built using TensorFlow and deployed with Streamlit.

------------------------------------------------------------

🚀 DEPLOYED APP
---------------
You can try the live app here:

[https://tarushivasishth-fashion-assistant-app-i8rhd4.streamlit.app/]
------------------------------------------------------------

📦 FEATURES
----------
- Upload an image of a clothing item.
- Download the image dataset on running the app without the need for manual installation.
- Extract visual features using ResNet50.
- Recommend relevant outfit combinations based on a dataset.
- User-friendly interface powered by Streamlit.

------------------------------------------------------------

🧰 TECH STACK
------------
- Frontend/UI: Streamlit
- Model: TensorFlow with ResNet50
- Backend Utilities: NumPy, Pandas, Pillow, gdown
- Deployment: Streamlit Cloud

------------------------------------------------------------

🛠️ SETUP INSTRUCTIONS (RUN LOCALLY)
-----------------------------------

1. Clone the Repository:
   git clone https://github.com/tarushivasishth/Fashion-Assistant.git
   cd fashion-assistant

2. Create and Activate a Virtual Environment:
   python -m venv venv
   On macOS/Linux: source venv/bin/activate
   On Windows: venv\Scripts\activate

3. Install the Required Packages:
   pip install -r requirements.txt

4. Run the Streamlit App:
   streamlit run app.py

------------------------------------------------------------

📁 PROJECT STRUCTURE
--------------------

fashion-assistant/
│
├── app.py                 # Main Streamlit app script
├── utils.py               # Utility functions (load model, extract features, etc.)
├── requirements.txt       # All Python dependencies
├── images/                # (Optional) Folder for sample images
└── README.txt             # This documentation

------------------------------------------------------------

🙌 ACKNOWLEDGEMENTS
-------------------
- TensorFlow and Keras for model building
- Streamlit for rapid web deployment
- Google Drive + gdown for model loading
- ResNet50 for image feature extraction
