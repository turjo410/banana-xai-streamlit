## 🍌 Project Resources & Live Demo

<div align="center">

### 🚀 **Live Streamlit App**

[![Launch App](https://img.shields.io/badge/🚀_Try_Live_Demo-Streamlit_App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://bananaapp.streamlit.app)

**[🎉 Open Banana XAI Classification App](https://bananaapp.streamlit.app)**

*Interactive banana classification with explainable AI visualizations*

---

### 📁 **Download Complete Project**

[![Download Project](https://img.shields.io/badge/📦_Download_Project-Google_Drive-4285F4?style=for-the-badge&logo=googledrive&logoColor=white)](https://drive.google.com/file/d/1FFp5SMLxZhq_Oiyypy7fOpP2YSOBZ7xb/view?usp=sharing)

**[📂 Get Full Source Code & Resources](https://drive.google.com/file/d/1FFp5SMLxZhq_Oiyypy7fOpP2YSOBZ7xb/view?usp=sharing)**

*Complete project files, weights, and documentation for local setup*

</div>

---

## 📚 Kaggle Notebooks & Model Weights

<div align="center">

### 🔬 Complete Model Development Journey

</div>

<table>
<tr>
<td align="center" width="20%">
<a href="https://www.kaggle.com/code/turjo410/banana-cnn-notebook-1-custom-cnn">
<img src="https://img.shields.io/badge/Notebook%201-Custom%20CNN-orange?style=for-the-badge&logo=jupyter" alt="Custom CNN"/>
</a>
<br/><strong>Basic CNN Implementation</strong>
</td>
<td align="center" width="20%">
<a href="https://www.kaggle.com/code/turjo410/banana-cnn-notebook-2-efficientnet-b0-dense">
<img src="https://img.shields.io/badge/Notebook%202-EfficientNet-blue?style=for-the-badge&logo=jupyter" alt="EfficientNet"/>
</a>
<br/><strong>Transfer Learning</strong>
</td>
<td align="center" width="20%">
<a href="https://www.kaggle.com/code/turjo410/banana-cnn-notebook-3-vgg16-convnext">
<img src="https://img.shields.io/badge/Notebook%203-VGG16-green?style=for-the-badge&logo=jupyter" alt="VGG16"/>
</a>
<br/><strong>Classical Architectures</strong>
</td>
<td align="center" width="20%">
<a href="https://www.kaggle.com/code/turjo410/banana-cnn-notebook-4-vision-transformer">
<img src="https://img.shields.io/badge/Notebook%204-ViT-purple?style=for-the-badge&logo=jupyter" alt="Vision Transformer"/>
</a>
<br/><strong>Transformer Models</strong>
</td>
<td align="center" width="20%">
<a href="https://www.kaggle.com/code/turjo410/banana-cnn-notebook-5-xai-analysis">
<img src="https://img.shields.io/badge/Notebook%205-XAI-red?style=for-the-badge&logo=jupyter" alt="XAI Analysis"/>
</a>
<br/><strong>Explainable AI</strong>
</td>
</tr>
</table>

<div align="center">

### 💾 Pre-trained Weights

[![Download Weights](https://img.shields.io/badge/📥_Download-Model_Weights-4285F4?style=for-the-badge&logo=googledrive&logoColor=white)](https://drive.google.com/drive/folders/1am4mdy5jTWBEBRNWFjlWO_pSczga2bic?usp=sharing)

*All trained model weights available for download*

</div>

---



# 🍌 Banana Variety and Ripeness Classification with Explainable AI (XAI)

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

<div align="center">

### 🎯 Interactive Streamlit Web Application for Banana Classification with Explainable AI

**Course**: `CSE 366` - Artificial Intelligence  
**Institution**: Department of Computer Science  
**Deadline**: August 17, 2025, 23:59 (Asia/Dhaka)

</div>

---

## 📖 Assignment Overview

This Assignment is an **interactive Streamlit web application** that classifies banana varieties and ripeness levels using multiple deep learning models. The app provides comprehensive **Explainable AI (XAI)** visualizations including Grad-CAM, Grad-CAM++, Eigen-CAM, Ablation-CAM, and LIME to help users understand model predictions.

## ✨ Key Features

- 🔄 **Multiple AI Models**: Choose from 12 pre-trained models (Custom CNN, EfficientNet, DenseNet, VGG16, ViT, DeiT)
- 🍌 **Dual Classification**: Classify banana variety (Sagor, Champa, Bangla, Sabri) and ripeness level (Green, Semi-ripe, Ripe, Overripe)
- 📊 **5 XAI Methods**: Generate visualizations from Grad-CAM, Grad-CAM++, Eigen-CAM, Ablation-CAM, and LIME
- 📱 **Professional UI**: A responsive and polished user interface built with Streamlit
- 📸 **Flexible Image Input**: Upload your own images or select from a pre-packaged set of samples
- 📈 **Detailed Results**: View Top-3 predictions with their corresponding confidence scores
- 💾 **Export Feature**: Download the original image and all generated explanations as a single ZIP file
- 🎨 **Interactive Interface**: Experience real-time analysis with clear progress indicators

## 🚀 Quick Start

### 1. Clone Repository

git clone https://github.com/your-username/banana-xai-classification.git
cd banana-xai-classificatio


### 2. Install Dependencies

pip install -r requirements.txt


### 3. Setup Model Weights

Place your trained model weights (`.pth` files) in the `weights/` directory:

weights/
├── CustomCNN_variety_best.pth
├── EfficientNet-B0_variety_classification_best.pth
├── custom_cnn_ripeness_best.pth
└── ... (other model weights)



### 4. Add Sample Images

Place sample images in the `assets/` directory:

assets/
├── variety_samples/
│ ├── Champa_Kola_57.jpg
│ ├── Sagor_Kola_39.jpg
│ └── ...
└── ripeness_samples/
├── Green_banana.jpg
├── Ripe_banana.jpg
└── ...



### 5. Run the App

streamlit run app.py




The app will open in your browser at `http://localhost:8501`.

## 👥 Group Members & Contributions

| Name | Student ID | Role | Contribution |
|------|------------|------|-------------|
| **Shahriar Khan** | 2022-3-60-016 | 🔧 Lead Developer & Architecture | 25% |
| **Tanvir Rahman** | 2022-3-60-134 | 🤖 Model Training & Optimization | 25% |
| **Khalid Mahmud Joy** | 2022-3-60-149 | 💻 App Development & UI/UX | 25% |
| **Rifah Tamanna** | 2022-3-60-159 | 📝 Documentation & Testing | 25% |

### 🎯 Individual Contributions

<details>
<summary><b>Shahriar Khan (Lead Developer)</b></summary>

- Assignment architecture design and coordination
- XAI integration (Grad-CAM, LIME implementations)
- Error handling and debugging
- Code optimization and performance tuning

</details>

<details>
<summary><b>Tanvir Rahman (Model Trainer)</b></summary>

- Deep learning model training and validation
- Model architecture selection and hyperparameter tuning
- Weight file preparation and model evaluation
- Performance benchmarking

</details>

<details>
<summary><b>Khalid Mahmud Joy (App Developer)</b></summary>

- Streamlit interface development
- UI/UX design and responsive layouts
- Interactive components and visualizations
- Frontend functionality implementation

</details>

<details>
<summary><b>Rifah Tamanna (Documentation & Testing)</b></summary>

- README and user manual creation
- Application testing and quality assurance
- Bug reporting and fixing coordination
- Screenshots and demo preparation

</details>

## 🏗️ Assignment Structure

banana-xai-classification/
├── app.py # Main Streamlit application
├── config.py # Model configurations
├── models.py # Model architectures
├── utils.py # Utility functions and XAI methods
├── requirements.txt # Python dependencies
├── README.md # This file
├── USER_MANUAL.md # Detailed user guide
├── weights/ # Model weight files (.pth)
├── assets/ # Sample images
│ ├── variety_samples/
│ └── ripeness_samples/



## 🔧 Technical Specifications

### Supported Models
- **Custom CNN**: 6 convolutional layers + 2 fully connected layers
- **EfficientNet-B0**: Transfer learning approach
- **DenseNet121**: Architecture with dense connectivity patterns
- **VGG16**: Classical CNN architecture
- **ViT-Base-16**: Vision Transformer
- **DeiT-Small-16**: Data-efficient Image Transformer

### Classification Tasks
1. **Variety Classification**: 4 classes (Sagor, Champa, Bangla, Sabri Kola)
2. **Ripeness Detection**: 4 classes (Green, Semi-ripe, Ripe, Overripe)

### XAI Methods
- **Grad-CAM**: Gradient-weighted Class Activation Mapping
- **Grad-CAM++**: Improved Grad-CAM with better localization
- **Eigen-CAM**: Eigen decomposition-based method
- **Ablation-CAM**: Ablation-based explanations
- **LIME**: Local Interpretable Model-agnostic Explanations

## ✅ Requirements Checklist

### 📱 Functional Requirements (Met)

- ✅ **FR-1. Model Selection UI**: Dropdown with 12 available models and metadata display
- ✅ **FR-2. Image Input**: Supports JPG/PNG uploads and sample image selection
- ✅ **FR-3. Prediction Display**: Shows Top-3 predictions with confidence scores
- ✅ **FR-4. XAI Explanations**: Implements all 5 required methods in a side-by-side grid
- ✅ **FR-5. UI & Export**: Professional layout with sidebar controls and ZIP download functionality

### 🎨 Non-Functional Requirements (Met)

- ✅ **NFR-1. Clarity**: Clean labels, captions, and legends for all UI elements
- ✅ **NFR-2. Consistency**: The same prediction is shown across all XAI methods
- ✅ **NFR-3. Reproducibility**: Clean startup with a detailed README file
- ✅ **NFR-4. Graceful Handling**: Friendly error messages for missing weights or invalid inputs

## 📦 Dependencies

streamlit>=1.28.0
torch>=2.0.0
torchvision>=0.15.0
Pillow>=9.5.0
numpy>=1.24.0
matplotlib>=3.7.0
plotly>=5.15.0
pytorch-grad-cam>=1.4.8
lime>=0.2.0.1
scikit-image>=0.21.0
timm>=0.9.0
transformers>=4.30.0
opencv-python>=4.5.0
tqdm>=4.62.0


## 🚨 Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| **App won't start** | Check Python version (3.8+), install requirements |
| **Model not loading** | Verify weight files are in `weights/` directory |
| **Poor predictions** | Use clear, high-quality images with banana as main subject |
| **XAI methods failing** | Some failure is normal; 3-4 working methods is acceptable |

## 🎯 Demo Checklist

- [x] App launches successfully
- [x] Model selection dropdown visible and functional
- [x] Model metadata displayed correctly
- [x] Image upload and sample selection works
- [x] Top-3 predictions with probabilities shown
- [x] All 5 XAI explanations generate properly
- [x] Explanations clearly labeled side-by-side
- [x] Download option available and working

## 🙏 Acknowledgments

- **CSE 366 - Artificial Intelligence Course**
- **Department of Computer Science**
- **Course Instructors** for their guidance and requirements
- The **Open Source Community** for providing excellent libraries

## 📞 Contact

For questions or issues, please contact the team lead:
- **Shahriar Khan** - `2022-3-60-016`
- **Email**: [turjo410@gmail.com](mailto:turjo410@gmail.com)

---

<div align="center">

*Last Updated: August 17, 2025*

**🍌 Happy Analyzing! ✨**

</div>



