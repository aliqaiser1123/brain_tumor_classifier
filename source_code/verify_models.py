import torch
import utils
from PIL import Image
import os
import numpy as np

def test_model(model_name, filename):
    print(f"Testing {model_name}...")
    if not os.path.exists(filename):
        print(f"FAILED: {filename} not found.")
        return

    try:
        model = utils.load_model(model_name, filename)
        if model:
            print(f"SUCCESS: {model_name} loaded.")
            
            # Create dummy image
            dummy_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
            
            # Test Prediction
            cls, conf, prob = utils.predict_image(model, dummy_img)
            print(f"Prediction: {cls} ({conf:.2f})")
            
            # Test Grad-CAM
            print("Testing Grad-CAM...")
            heatmap = utils.generate_gradcam(model, dummy_img, model_name)
            if heatmap is not None:
                print("SUCCESS: Grad-CAM generated.")
            else:
                print("FAILED: Grad-CAM returned None.")
        else:
            print(f"FAILED: Model returned None.")
    except Exception as e:
        print(f"ERROR: {e}")
    print("-" * 20)

if __name__ == "__main__":
    test_model("ResNet18", "resnet18_brain_tumor.pth")
    test_model("EfficientNet-B0", "efficientnet_b0_brain_tumor.pth")
