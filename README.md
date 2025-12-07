# AI_image_Classifier

📘 AI Image Classifier
A simple and powerful image classification web app built using Streamlit, TensorFlow (MobileNetV2), OpenCV, and Pillow.
Upload any image, and the AI model will instantly predict what’s inside it.

🚀 Features
🔍 Classify images using MobileNetV2 (ImageNet-trained)
🖼️ Upload JPG, JPEG, PNG images
⚙️ Preprocessing via OpenCV (cv2)
📊 Displays top-3 predictions with confidence scores
⚡ Fast model loading with Streamlit caching
🌐 Deployable to Streamlit Cloud

📂 Project Structure
ai-image-classifier/
│── main.py
│── requirements.txt
│── README.md

🛠️ Installation & Setup
1️⃣ Clone the repository
git clone <your-repo-link>
cd ai-image-classifier
2️⃣ Install dependencies
Use the included requirements.txt:
pip install -r requirements.txt
3️⃣ Run the Streamlit App
streamlit run main.py


Your app will open in the browser at:
👉 http://localhost:8501

📦 Requirements
Your requirements.txt should include:

streamlit
numpy
opencv-python-headless
Pillow
tensorflow

If deploying on Streamlit Cloud, you can use tensorflow-cpu instead of tensorflow for faster installs.

🧠 How It Works
User uploads an image
Image is converted to RGB (to avoid alpha-channel issues)
Image is resized to 224 × 224 using OpenCV
Preprocessing is applied using preprocess_input()
MobileNetV2 predicts top 3 labels
Predictions & confidence scores are shown

🧩 Code Highlights
Image Preprocessing (cv2-based)
def preprocess_image(image):
    image = image.convert("RGB")
    img = np.array(image)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img

Prediction
predictions = model.predict(processed_image)
decoded = decode_predictions(predictions, top=3)[0]

🎯 Model Used
MobileNetV2
Pre-trained on ImageNet (1000 classes)
Lightweight & fast → ideal for live predictions

🌐 Deploying to Streamlit Cloud
Push your project to GitHub
Go to share.streamlit.io

Select your repo
Set:
Main file: main.py
Python version: 3.11 (recommended)
Dependencies: picked from requirements.txt
That’s it! App will deploy automatically.

📸 Screenshots (Optional)
Add screenshots of your app interface here if you want.

🤝 Contributing
Pull requests are welcome!
If you want enhancements (Grad-CAM heatmaps, custom models, multi-page UI), feel free to open an issue.
