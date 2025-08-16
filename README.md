# 🍌 Banana AI Explorer - XAI Dashboard

A comprehensive Streamlit application for banana classification using multiple deep learning models with explainable AI (XAI) visualizations.

## Features

- **Multiple Model Support**: Custom CNN, EfficientNet-B0, DenseNet121, VGG16, Vision Transformers (ViT & DeiT)
- **Dual Classification Tasks**: Banana variety classification and ripeness detection
- **5 XAI Methods**: Grad-CAM, Grad-CAM++, Eigen-CAM, Ablation-CAM, and LIME
- **Interactive UI**: Modern, responsive interface with tabs and expandable sections
- **Download Capability**: Export all visualizations as ZIP file
- **Robust Error Handling**: Graceful fallbacks and user-friendly error messages

## Installation

1. Clone the repository and navigate to project directory
2. Create virtual environment: `python -m venv venv`
3. Activate environment: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (macOS/Linux)
4. Install dependencies: `pip install -r requirements.txt`

## Setup

1. Copy all your .pth model weights to the `weights/` directory
2. Add sample images to `assets/variety_samples/` and `assets/ripeness_samples/`
3. Run: `streamlit run app.py`
4. Open browser at `http://localhost:8501`

## Usage

1. Select a model from the dropdown menu
2. Upload an image or choose from samples
3. View predictions and XAI visualizations
4. Download results as needed

## Team Members & Contributions

- [Member 1]: Model development, XAI implementation
- [Member 2]: UI/UX design, Streamlit development
- [Member 3]: Testing, documentation, deployment

## License

This project is for educational purposes (CSE 366 - Artificial Intelligence).
#
