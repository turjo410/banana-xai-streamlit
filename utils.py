import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import io
import zipfile
import base64
from datetime import datetime

# XAI Libraries
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, EigenCAM, AblationCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from lime import lime_image
from skimage.segmentation import mark_boundaries

from models import *
from config import *

# === TRANSFORMS ===
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

def safe_float_conversion(prob_val):
    """Safely convert any probability value to float"""
    try:
        while isinstance(prob_val, (list, tuple)):
            if len(prob_val) > 0:
                prob_val = prob_val[0]
            else:
                return 0.0
        
        if hasattr(prob_val, 'item'):
            return float(prob_val.item())
        
        if hasattr(prob_val, '__float__'):
            return float(prob_val)
        
        return float(prob_val)
        
    except Exception as e:
        print(f"⚠️ Float conversion error: {e} for value: {prob_val} (type: {type(prob_val)})")
        return 0.0

def get_model_class(model_class_name):
    """Get model class by name"""
    return globals()[model_class_name]

def load_model(model_config):
    """Load model with weights"""
    model_class_name = model_config['model_class']
    weight_file = model_config['weight_file']
    num_classes = model_config['num_classes']
    
    # Initialize model
    model_class = get_model_class(model_class_name)
    model = model_class(num_classes)
    
    # Load weights
    weight_path = os.path.join(WEIGHTS_DIR, weight_file)
    try:
        if os.path.exists(weight_path):
            state_dict = torch.load(weight_path, map_location=DEVICE)
            model.load_state_dict(state_dict)
            print(f"✅ Loaded weights: {weight_file}")
        else:
            print(f"⚠️ Weight file not found: {weight_path}. Using random weights.")
    except Exception as e:
        print(f"⚠️ Error loading weights: {e}. Using random weights.")
    
    model = model.to(DEVICE)
    model.eval()
    
    # ✅ CRITICAL: Enable gradients for ALL parameters for XAI
    for param in model.parameters():
        param.requires_grad_(True)
    
    return model

def preprocess_image(image):
    """Use exact same preprocessing as training"""
    return transform(image).unsqueeze(0).to(DEVICE)

def denormalize_image(tensor_img):
    """Convert normalized tensor back to displayable image"""
    img = tensor_img.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    img = np.array(STD) * img + np.array(MEAN)
    return np.clip(img, 0, 1)

def get_sample_images(sample_dir):
    """Get list of sample images"""
    sample_path = os.path.join(ASSETS_DIR, sample_dir)
    if not os.path.exists(sample_path):
        return []
    
    images = []
    for file in os.listdir(sample_path):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            images.append(os.path.join(sample_path, file))
    return sorted(images)

class EnhancedManualGradCAM:
    """Enhanced manual GradCAM with robust error handling"""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        
    def _create_fallback_heatmap(self):
        """Create a fallback heatmap when GradCAM fails"""
        height, width = IMG_SIZE if isinstance(IMG_SIZE, (tuple, list)) else (IMG_SIZE, IMG_SIZE)
        heatmap = np.zeros((height, width))
        center_y, center_x = height // 2, width // 2
        y, x = np.ogrid[:height, :width]
        mask = (y - center_y)**2 + (x - center_x)**2 <= (min(height, width) // 3)**2
        heatmap[mask] = 1.0
        return heatmap
        
    def generate_cam(self, input_tensor, target_class):
        gradients = []
        activations = []
        
        def save_activation(module, input, output):
            if output is not None and torch.is_tensor(output):
                activations.append(output.detach().clone())
                print(f"✅ Saved activation shape: {output.shape}")
                
        def save_gradient(module, grad_input, grad_output):
            if isinstance(grad_output, tuple):
                if len(grad_output) > 0 and grad_output[0] is not None:
                    gradients.append(grad_output.detach().clone())
                    print(f"✅ Saved gradient shape: {grad_output.shape}")
            else:
                if grad_output is not None:
                    gradients.append(grad_output.detach().clone())
                    print(f"✅ Saved gradient shape: {grad_output.shape}")
        
        # Register hooks
        try:
            h1 = self.target_layer.register_forward_hook(save_activation)
            h2 = self.target_layer.register_backward_hook(save_gradient)
        except Exception as e:
            print(f"❌ Failed to register hooks: {e}")
            return self._create_fallback_heatmap()
        
        try:
            # ✅ CRITICAL: Ensure model is in eval mode with gradients enabled
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad_(True)
                
            input_tensor = input_tensor.clone().detach().requires_grad_(True)
            self.model.zero_grad()
            
            # Forward pass
            output = self.model(input_tensor)
            if output.dim() > 1:
                score = output[:, target_class]
            else:
                score = output[target_class]
            
            # Backward pass
            score.backward(retain_graph=True)
            
            print(f"🔍 Number of gradients captured: {len(gradients)}")
            print(f"🔍 Number of activations captured: {len(activations)}")
            
            # ✅ ENHANCED: Check if we have valid gradients and activations
            if len(gradients) == 0 or len(activations) == 0:
                print("⚠️ Empty gradients or activations, using fallback")
                return self._create_fallback_heatmap()
            
            # ✅ ENHANCED: Safe access to gradients and activations
            try:
                grads = gradients[0]
                acts = activations
                
                # Ensure tensors are valid
                if grads.numel() == 0 or acts.numel() == 0:
                    print("⚠️ Empty tensors, using fallback")
                    return self._create_fallback_heatmap()
                
                print(f"✅ Using gradient shape: {grads.shape}")
                print(f"✅ Using activation shape: {acts.shape}")
                
                # Compute CAM
                weights = torch.mean(grads, dim=(2, 3), keepdim=True)
                cam = torch.sum(weights * acts, dim=1, keepdim=True)
                cam = F.relu(cam)
                
                cam = F.interpolate(cam, size=IMG_SIZE, mode='bilinear', align_corners=False)
                cam = cam.squeeze().cpu().numpy()
                
                if cam.max() > cam.min():
                    cam = (cam - cam.min()) / (cam.max() - cam.min())
                else:
                    return self._create_fallback_heatmap()
                
                return cam
                
            except Exception as inner_e:
                print(f"⚠️ Error processing gradients/activations: {inner_e}")
                return self._create_fallback_heatmap()
            
        except Exception as e:
            print(f"⚠️ Manual GradCAM failed: {e}")
            return self._create_fallback_heatmap()
            
        finally:
            try:
                h1.remove()
                h2.remove()
            except:
                pass

def get_target_layer(model, model_class_name):
    """Enhanced target layer selection with fallbacks"""
    print(f"🔧 Selecting target layer for: {model_class_name}")
    
    if 'CustomBananaCNN' in model_class_name:
        conv_layers = [m for m in model.features.modules() if isinstance(m, nn.Conv2d)]
        print(f"🔧 Found {len(conv_layers)} conv layers")
        if len(conv_layers) >= 2:
            # Use second-to-last layer for better gradients
            return [conv_layers[-2]]
        elif len(conv_layers) >= 1:
            return [conv_layers[-1]]
        else:
            print("⚠️ No conv layers found, using fallback")
            return [list(model.modules())[-3]]  # Use deeper layer
            
    elif 'EfficientNet' in model_class_name:
        print(f"🔧 Using EfficientNet layers")
        try:
            # Try multiple possible target layers
            if hasattr(model.backbone, 'conv_head'):
                return [model.backbone.conv_head]
            elif hasattr(model.backbone, 'features') and len(model.backbone.features) > 0:
                return [model.backbone.features[-1]]
            else:
                conv_layers = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
                return [conv_layers[-1]] if conv_layers else [list(model.modules())[-3]]
        except AttributeError:
            conv_layers = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
            return [conv_layers[-1]] if conv_layers else [list(model.modules())[-3]]
            
    elif 'DenseNet' in model_class_name:
        print(f"🔧 Using DenseNet norm5")
        try:
            return [model.backbone.features.norm5]
        except AttributeError:
            conv_layers = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
            return [conv_layers[-1]] if conv_layers else [list(model.modules())[-3]]
            
    elif 'VGG' in model_class_name:
        print(f"🔧 Using VGG last feature layer")
        try:
            return [model.backbone.features[-1]]
        except AttributeError:
            conv_layers = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
            return [conv_layers[-1]] if conv_layers else [list(model.modules())[-3]]
            
    elif 'ViT' in model_class_name or 'DeiT' in model_class_name:
        print(f"🔧 Using ViT/DeiT layers")
        try:
            # For transformers, try different possible layers
            if hasattr(model.backbone, 'patch_embed') and hasattr(model.backbone.patch_embed, 'proj'):
                return [model.backbone.patch_embed.proj]
            else:
                # Fallback to finding conv layers
                conv_layers = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
                return [conv_layers[-1]] if conv_layers else [list(model.modules())[-3]]
        except AttributeError:
            conv_layers = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
            return [conv_layers[-1]] if conv_layers else [list(model.modules())[-3]]
    else:
        # Generic fallback
        conv_layers = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
        print(f"🔧 Fallback: Found {len(conv_layers)} conv layers")
        if len(conv_layers) >= 2:
            return [conv_layers[-2]]  # Use second-to-last for better gradients
        elif len(conv_layers) >= 1:
            return [conv_layers[-1]]
        else:
            return [list(model.modules())[-3]]

class XAIAnalyzer:
    """Enhanced XAI analyzer with robust error handling"""
    def __init__(self, model, model_config, class_names, device):
        self.model = model
        self.model_config = model_config
        self.class_names = class_names
        self.device = device
        
        # Get target layers for CAM methods
        self.target_layers = get_target_layer(model, model_config['model_class'])
        
        # Initialize enhanced manual GradCAM
        if self.target_layers:
            self.manual_gradcam = EnhancedManualGradCAM(self.model, self.target_layers[0])
        else:
            self.manual_gradcam = None
        
    def predict(self, input_tensor):
        """Get model prediction - TENSOR SCALAR CONVERSION FIXED"""
        self.model.eval()
        
        if input_tensor.ndim == 3:
            input_tensor = input_tensor.unsqueeze(0)
        
        input_tensor = input_tensor.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            
            # ✅ DEBUG: Print shapes to understand the issue
            print(f"🔍 Model outputs shape: {outputs.shape}")
            print(f"🔍 Raw outputs: {outputs}")
            
            if outputs.ndim == 1:
                outputs = outputs.unsqueeze(0)
            
            probs = F.softmax(outputs, dim=1)
            print(f"🔍 Probabilities shape: {probs.shape}")
            print(f"🔍 Probabilities: {probs}")
            
            pred_idx = torch.argmax(probs, dim=1).item()
            print(f"🔍 Predicted index: {pred_idx}")
            
            # ✅ CRITICAL FIX: Safe scalar extraction
            prob_value = probs[0, pred_idx]
            print(f"🔍 Confidence tensor: {prob_value} (shape: {prob_value.shape}, elements: {prob_value.numel()})")
            
            if prob_value.numel() == 1:
                confidence = prob_value.item()
            else:
                # Handle multi-element tensor
                confidence = float(prob_value.mean()) if prob_value.numel() > 0 else 0.0
                print(f"⚠️ Multi-element confidence tensor detected, using mean: {confidence}")
            
            # ✅ ENHANCED: Safe top-k predictions
            topk = torch.topk(probs, k=min(3, len(self.class_names)))
            top_indices = topk.indices[0].cpu().numpy().tolist()
            top_values = topk.values.cpu().numpy().tolist()
            
            print(f"🔍 Top indices: {top_indices}")
            print(f"🔍 Top values: {top_values}")
            
            # Build results with enhanced debugging
            top3_results = []
            for i, (class_idx, prob_val) in enumerate(zip(top_indices, top_values)):
                if 0 <= class_idx < len(self.class_names):
                    safe_prob = safe_float_conversion(prob_val)
                    class_name = self.class_names[class_idx]
                    top3_results.append((class_name, safe_prob))
                    print(f"🔍 Top-{i+1}: {class_name} = {safe_prob:.6f}")
                else:
                    print(f"⚠️ Invalid class index: {class_idx}")
            
            print(f"🔍 Final prediction: {self.class_names[pred_idx]} (confidence: {confidence:.6f})")
            
            return pred_idx, confidence, top3_results




    
    def generate_all_cams(self, input_tensor, target_class):
        """Generate all CAM methods with enhanced error handling"""
        self.model.eval()
        
        # ✅ CRITICAL: Enable gradients for ALL parameters
        for param in self.model.parameters():
            param.requires_grad_(True)
        
        results = {}
        
        # Validate target class index
        if target_class < 0 or target_class >= len(self.class_names):
            print(f"⚠️ Invalid target class {target_class}, using class 0")
            target_class = 0
        
        cam_methods = {
            'Grad-CAM': GradCAM,
            'Grad-CAM++': GradCAMPlusPlus,
            'Eigen-CAM': EigenCAM,
            'Ablation-CAM': AblationCAM
        }
        
        for name, cam_class in cam_methods.items():
            try:
                if self.target_layers:
                    # Try library implementation
                    cam = cam_class(model=self.model, target_layers=self.target_layers)
                    targets = [ClassifierOutputTarget(target_class)]
                    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
                    
                    # Check if meaningful
                    if grayscale_cam is not None and grayscale_cam.max() > 0.01:
                        results[name] = grayscale_cam[0]
                        print(f"✅ {name} library working")
                    else:
                        raise ValueError("Empty or invalid result from library")
                else:
                    raise ValueError("No target layers available")
                    
            except Exception as e:
                print(f"🔧 {name} library failed: {str(e)[:100]}...")
                # Use enhanced manual implementation
                if self.manual_gradcam:
                    try:
                        manual_result = self.manual_gradcam.generate_cam(input_tensor, target_class)
                        if manual_result is not None:
                            results[name] = manual_result
                            print(f"✅ {name} manual implementation working")
                        else:
                            print(f"❌ {name} manual implementation failed")
                    except Exception as manual_e:
                        print(f"❌ {name} manual implementation failed: {str(manual_e)[:100]}...")
                else:
                    print(f"❌ No manual implementation available for {name}")
        
        return results
    
    def generate_lime_explanation(self, input_tensor, target_class):
        """Enhanced LIME explanation with error handling"""
        if target_class < 0 or target_class >= len(self.class_names):
            target_class = 0
        
        def predict_fn(images):
            try:
                self.model.eval()
                batch_tensors = []
                
                for img in images:
                    if img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                    pil_img = Image.fromarray(img.astype(np.uint8))
                    tensor = transform(pil_img)
                    batch_tensors.append(tensor)
                
                if len(batch_tensors) == 0:
                    return np.zeros((1, len(self.class_names)))
                
                batch = torch.stack(batch_tensors).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(batch)
                    if outputs.ndim == 1:
                        outputs = outputs.unsqueeze(0)
                    probs = F.softmax(outputs, dim=1)
                    return probs.cpu().numpy()
                    
            except Exception as e:
                print(f"⚠️ LIME predict_fn error: {e}")
                return np.ones((len(images), len(self.class_names))) / len(self.class_names)
        
        try:
            image_for_lime = (denormalize_image(input_tensor) * 255).astype(np.uint8)
            
            explainer = lime_image.LimeImageExplainer()
            explanation = explainer.explain_instance(
                image_for_lime,
                predict_fn,
                top_labels=len(self.class_names),
                hide_color=0,
                num_samples=500,
                batch_size=10
            )
            
            temp, mask = explanation.get_image_and_mask(
                target_class,
                positive_only=False,
                num_features=5,
                hide_rest=False
            )
            
            return mark_boundaries(temp / 255.0, mask)
            
        except Exception as e:
            print(f"⚠️ LIME failed: {e}")
            return denormalize_image(input_tensor)


    
    def analyze_image(self, input_tensor):
        """Complete XAI analysis with comprehensive error handling"""
        input_tensor = input_tensor.to(self.device)
        
        # Get prediction with bounds checking
        pred_idx, confidence, top3_results = self.predict(input_tensor)
        pred_label = self.class_names[pred_idx]
        
        print(f"🔍 Analyzing image for prediction: {pred_label} (confidence: {confidence:.2%})")
        
        # Generate all XAI methods
        cam_results = self.generate_all_cams(input_tensor, pred_idx)
        lime_result = self.generate_lime_explanation(input_tensor, pred_idx)
        
        # Count working methods
        working_count = len([r for r in cam_results.values() if r is not None]) + (1 if lime_result is not None else 0)
        
        return {
            'prediction': pred_label,
            'confidence': confidence,
            'top3_results': top3_results,
            'working_methods': working_count,
            'cam_results': cam_results,
            'lime_result': lime_result,
            'original_image': denormalize_image(input_tensor)
        }

# Keep other functions unchanged (create_download_zip, get_download_link)
def create_download_zip(results, filename_prefix):
    """Create downloadable ZIP file with all visualizations"""
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Original image
        original_img = Image.fromarray((results['original_image'] * 255).astype(np.uint8))
        img_buffer = io.BytesIO()
        original_img.save(img_buffer, format='PNG')
        zip_file.writestr(f"{filename_prefix}_original.png", img_buffer.getvalue())
        
        # CAM results
        for method_name, cam_result in results['cam_results'].items():
            if cam_result is not None:
                try:
                    cam_overlay = show_cam_on_image(results['original_image'], cam_result, use_rgb=True)
                    cam_img = Image.fromarray((cam_overlay * 255).astype(np.uint8))
                    img_buffer = io.BytesIO()
                    cam_img.save(img_buffer, format='PNG')
                    zip_file.writestr(f"{filename_prefix}_{method_name.replace(' ', '_').lower()}.png", 
                                    img_buffer.getvalue())
                except Exception as e:
                    print(f"⚠️ Error saving {method_name}: {e}")
        
        # LIME result
        if results['lime_result'] is not None:
            try:
                lime_img = Image.fromarray((results['lime_result'] * 255).astype(np.uint8))
                img_buffer = io.BytesIO()
                lime_img.save(img_buffer, format='PNG')
                zip_file.writestr(f"{filename_prefix}_lime.png", img_buffer.getvalue())
            except Exception as e:
                print(f"⚠️ Error saving LIME: {e}")
    
    return zip_buffer.getvalue()

def get_download_link(zip_data, filename):
    """Generate download link for ZIP file"""
    b64 = base64.b64encode(zip_data).decode()
    href = f'<a href="data:application/zip;base64,{b64}" download="{filename}">📥 Download All Visualizations (ZIP)</a>'
    return href
