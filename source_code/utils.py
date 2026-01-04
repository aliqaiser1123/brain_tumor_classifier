import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import numpy as np
import cv2
from PIL import Image
import torch.nn.functional as F

# Constants
CLASSES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

def load_model(model_name, model_path):
    """
    Loads the specified model with weights from model_path.
    """
    try:
        if model_name == 'ResNet18':
            model = models.resnet18(weights=None)
            # Adjust final layer for 4 classes
            model.fc = nn.Linear(model.fc.in_features, 4)
        elif model_name == 'EfficientNet-B0':
            model = models.efficientnet_b0(weights=None)
            # Adjust final layer for 4 classes
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, 4)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model = model.to(DEVICE)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model {model_name}: {e}")
        return None

def predict_image(model, image):
    """
    Predicts the class of the image using the model.
    """
    transform = get_transforms()
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)
        
    return CLASSES[predicted_class.item()], confidence.item(), probabilities.cpu().numpy()[0]

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hook the target layer
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx=None):
        self.model.eval()
        output = self.model(x)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
            
        self.model.zero_grad()
        class_loss = output[0, class_idx]
        class_loss.backward()
        
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (x.shape[3], x.shape[2]))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam

def generate_gradcam(model, image, model_name):
    transform = get_transforms()
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    # Identify target layer based on model architecture
    if model_name == 'ResNet18':
        target_layer = model.layer4[-1]
    elif model_name == 'EfficientNet-B0':
        # For EfficientNet, usually the last convolutional block features
        target_layer = model.features[-1]
    else:
        return None

    grad_cam = GradCAM(model, target_layer)
    mask = grad_cam(img_tensor)
    
    return mask

def overlay_heatmap(image, mask):
    """
    Overlays the Grad-CAM heatmap on the original image.
    """
    img_np = np.array(image.convert('RGB'))
    img_np = cv2.resize(img_np, (224, 224))
    
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    img_float = np.float32(img_np) / 255
    
    cam = heatmap + img_float
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)
