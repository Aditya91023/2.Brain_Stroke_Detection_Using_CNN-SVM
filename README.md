
# 🧠 Brain Stroke Detection using CNN and SVM

This project aims to detect brain strokes using CT scan images with two different approaches:
- A deep learning model based on Convolutional Neural Networks (CNN)
- A traditional machine learning model using Support Vector Machine (SVM) with feature extraction

It also features a web-based interface built with Gradio for real-time predictions.

---

## 📁 Dataset

We’re using the **Brain Stroke CT Image Dataset** from Kaggle:

🔗 [Kaggle Dataset Link](https://www.kaggle.com/datasets/afridirahman/brain-stroke-ct-image-dataset)

The dataset contains two folders:
- `Stroke` (images with brain stroke)
- `Normal` (images without stroke)

After downloading, organize it like this:

```
Brain_Data_Organised/
├── Stroke/
└── Normal/
```

---

## ✨ What This Project Does

- Trains a CNN on brain CT images.
- Trains an SVM using features extracted from MobileNetV2.
- Compares both models using metrics like Accuracy, Precision, Recall, F1-Score, and ROC AUC.
- Provides visualizations to help understand model performance.
- Includes a Gradio UI for uploading CT images and getting real-time predictions.

---

## 📊 Visualizations Included

- **Training Accuracy & Loss** for the CNN.
- **ROC AUC Curve** to compare how well each model distinguishes between classes.
- **Bar Charts** comparing CNN and SVM on multiple performance metrics.
- **Confusion Matrix** (optional to add if needed).

---

## 🔍 Model Insights

| Metric     | CNN (Deep Learning) | SVM (with MobileNetV2 Features) |
|------------|---------------------|----------------------------------|
| Accuracy   | ✅ High              | ✅ Competitive                   |
| Speed      | ⏳ Slower to train   | ⚡ Very fast                     |
| Complexity | 🧠 Deep model        | 🧮 Lightweight linear model      |

---

## 🧪 Try It Out With Gradio

Once the models are trained and saved, run the Gradio app:

```python
iface.launch()
```

Upload a CT image and select either **CNN** or **SVM** to get a prediction instantly.

---

## 🛠 Tech Stack

- Python
- TensorFlow / Keras
- scikit-learn
- OpenCV
- Matplotlib & Seaborn
- Gradio
- Google Colab compatible

---

## 📦 Folder Overview

```
.
├── Brain_Data_Organised/
│   ├── Stroke/
│   └── Normal/
├── cnn_stroke_model.h5
├── svm_stroke_model.pkl
├── brain_stroke_cnn_svm.ipynb
├── README.md
└── requirements.txt
```

---

## ✅ How to Run

1. Download the dataset from Kaggle.
2. Place it in `Brain_Data_Organised/Stroke` and `.../Normal`.
3. Open the `.ipynb` notebook in Google Colab.
4. Run all cells to train models and visualize results.
5. Use Gradio to make predictions.

---

## 👤 Author

**Your Name Here**  
📧 your.email@example.com  
🐙 GitHub: [yourusername](https://github.com/yourusername)

---

## 📄 License

MIT License. Feel free to use and modify.
