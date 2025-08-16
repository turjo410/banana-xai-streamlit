import gdown
import os
from pathlib import Path
import streamlit as st

# Map all your model weights from the screenshot
MODEL_WEIGHTS = {
    "🚀 ConvNeXt-Tiny (Ripeness)": "ConvNeXt-Tiny_ripeness_detection_best.pth",
    "🍌 ConvNeXt-Tiny (Variety)": "ConvNeXt-Tiny_variety_classification_best.pth", 
    "🚀 Custom CNN (Ripeness)": "custom_cnn_ripeness_best.pth",
    "🍌 Custom CNN (Variety)": "CustomCNN_variety_best.pth",
    "🚀 DeiT-Small-16 (Ripeness)": "DeiT-Small-16_ripeness_detection_best.pth",
    "🍌 DeiT-Small-16 (Variety)": "DeiT-Small-16_variety_classification_best.pth",
    "🚀 DenseNet121 (Ripeness)": "DenseNet121_ripeness_detection_best.pth", 
    "🍌 DenseNet121 (Variety)": "DenseNet121_variety_classification_best.pth",
    "🚀 EfficientNet-B0 (Ripeness)": "EfficientNet-B0_ripeness_detection_best.pth",
    "🍌 EfficientNet-B0 (Variety)": "EfficientNet-B0_variety_classification_best.pth",
    "🚀 VGG16 (Ripeness)": "VGG16_ripeness_detection_best.pth",
    "🍌 VGG16 (Variety)": "VGG16_variety_classification_best.pth", 
    "🚀 ViT-Base-16 (Ripeness)": "ViT-Base-16_ripeness_detection_best.pth",
    "🍌 ViT-Base-16 (Variety)": "ViT-Base-16_variety_classification_best.pth",
}

# Google Drive File IDs - YOU NEED TO REPLACE THESE WITH YOUR ACTUAL IDs
GDRIVE_FILE_IDS = {
    "ConvNeXt-Tiny_ripeness_detection_best.pth": "1VZb-vYwIV-Wp9kCvXovChYd_ARpq9tSU",
    "ConvNeXt-Tiny_variety_classification_best.pth": "1X5ACqc3TAFEbOzlXtu47gp77zeNbrzyn",
    "custom_cnn_ripeness_best.pth": "1py3dAR-Q4PWM6WkJ5OhrdNOYTzndJMyn", 
    "CustomCNN_variety_best.pth": "1S5m76Bcd4Dg-Y6sKKd-NBT0Ya01qqNkG",
    "DeiT-Small-16_ripeness_detection_best.pth": "1T3FOtUyVlRsRBiuXDLwotQlloWVkMmE3",
    "DeiT-Small-16_variety_classification_best.pth": "1TqpmP4gBB7ElJvMXps59j_NeWWF1M9sG",
    "DenseNet121_ripeness_detection_best.pth": "1sAI92ZfjYWDVumcgCHZJQ98qe3XGZwWN",
    "DenseNet121_variety_classification_best.pth": "1O2hdu76zo4imGj5gNkuCT865QuRSi0cc",
    "EfficientNet-B0_ripeness_detection_best.pth": "1g14K2TxW1XfpRUAjsCNnSLcS6GsIn-II",
    "EfficientNet-B0_variety_classification_best.pth": "1NtyHshx4pyFNLkpU8PWIVl1kQ3xJ97Dn",
    "VGG16_ripeness_detection_best.pth": "10CVRO8dw944qV5WbzLfEkQGb2dmgot9Z",
    "VGG16_variety_classification_best.pth": "1W2lD7gIC2CUS0fXiDWHifXxegcjryksm",
    "ViT-Base-16_ripeness_detection_best.pth": "1LMRKfVzqDD-F2tKtisGjX6b9FF-3_6m9",
    "ViT-Base-16_variety_classification_best.pth": "1Dhv93sSFm_4udlENRF11sTsI5lgsiAdq",
}

@st.cache_resource
def download_model_weights():
    """Download all model weights from Google Drive on first run"""
    weights_dir = Path("weights")
    weights_dir.mkdir(exist_ok=True)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_models = len(MODEL_WEIGHTS)
    downloaded = 0
    
    for display_name, filename in MODEL_WEIGHTS.items():
        filepath = weights_dir / filename
        
        if not filepath.exists():
            file_id = GDRIVE_FILE_IDS.get(filename)
            
            if file_id and file_id.startswith("REPLACE"):
                st.error(f"❌ Please update Google Drive file ID for {filename}")
                continue
                
            if file_id:
                try:
                    status_text.text(f"🔄 Downloading {display_name}...")
                    url = f"https://drive.google.com/uc?id={file_id}"
                    gdown.download(url, str(filepath), quiet=True)
                    st.success(f"✅ Downloaded {display_name}")
                except Exception as e:
                    st.error(f"❌ Failed to download {display_name}: {str(e)}")
            else:
                st.warning(f"⚠️ No Google Drive ID found for {filename}")
        else:
            st.info(f"✅ {display_name} already exists")
            
        downloaded += 1
        progress_bar.progress(downloaded / total_models)
    
    status_text.text("🎉 All models ready!")
    return weights_dir

# Alternative: Download entire folder (EASIER METHOD)
@st.cache_resource
def download_weights_folder():
    """Download entire weights folder from Google Drive"""
    folder_id = "1am4mdy5jTWBEBRNWFjlWO_pSczga2bic"  # Your folder ID
    
    weights_dir = Path("weights")
    weights_dir.mkdir(exist_ok=True)
    
    with st.spinner("🔄 Downloading all model weights... This may take 5-10 minutes on first run!"):
        try:
            # Create a temporary download folder
            temp_dir = Path("temp_weights")
            temp_dir.mkdir(exist_ok=True)
            
            # Download the folder
            gdown.download_folder(
                f"https://drive.google.com/drive/folders/{folder_id}",
                output=str(temp_dir),
                quiet=False,
                use_cookies=False
            )
            
            # Move files to weights directory
            for file_path in temp_dir.glob("**/*.pth"):
                destination = weights_dir / file_path.name
                file_path.rename(destination)
            
            # Cleanup temporary directory
            import shutil
            shutil.rmtree(temp_dir)
            
            st.success("✅ All weights downloaded successfully!")
        except Exception as e:
            st.error(f"❌ Failed to download weights folder: {str(e)}")
    
    return weights_dir
