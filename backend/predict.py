import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import timm
from torch import nn
import joblib
import numpy as np
import os
import json
import re

from llm import run_fashion_llm
from extract_info import parse_llm_recommendations, save_recommendations_to_json, save_facial_features_to_json
import subprocess
from ml import predict_single_image

# Path to the trained facial feature model
facial_model_path = 'celeba_imbalance_aware_classifier.pth'

class AgeGenderModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b4', pretrained=True, num_classes=0, global_pool='avg')
        self.gender_head = nn.Sequential(
            nn.Linear(self.backbone.num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )
        self.age_head = nn.Sequential(
            nn.Linear(self.backbone.num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        feat = self.backbone(x)
        return torch.sigmoid(self.gender_head(feat)).squeeze(1), self.age_head(feat).squeeze(1)

# Model/scaler loading
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AgeGenderModel().to(device)
    model_path = "age_gender_model.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: '{model_path}'.")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.eval()

    scaler_path = 'age_scaler.pkl'
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found: '{scaler_path}'.")
    scaler = joblib.load(scaler_path)

except FileNotFoundError as e:
    print(f"CRITICAL ERROR: {e}")
    print("Please ensure 'age_gender_model.pth' and 'age_scaler.pkl' are in the same directory.")
    exit(1)
except Exception as e:
    print(f"CRITICAL ERROR loading age/gender model or scaler: {e}")
    exit(1)

transform = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor()
])

def process_image_for_recommendations(image_path, height, weight, occasion):
    """Process image for fashion recommendations using EXACT same logic as working test_my_pipeline.py"""
    try:
        # Load image exactly like the working test script
        import cv2
        cv_img = cv2.imread(image_path)
        if cv_img is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Convert BGR to RGB (same as test script: cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image (same as test script: Image.fromarray(img_rgb))
        pil_img = Image.fromarray(img_rgb)
        input_tensor = transform(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            gender_prob, age_scaled = model(input_tensor)

        age = int(scaler.inverse_transform([[age_scaled.item()]])[0][0])
        gender = "Male" if gender_prob.item() < 0.5 else "Female"

        print("\n=== PREDICTION RESULT ===")
        print(f"Gender: {gender} (Confidence: {gender_prob.item():.2f})")
        print(f"Age: {age} years")
        print(f"Height: {height} cm")
        print(f"Weight: {weight} kg")
        print(f"Occasion: {occasion}")

        # Calculate BMI
        height_m = height / 100.0
        bmi = weight / (height_m ** 2)
        print(f"BMI: {bmi:.2f}")

        # --- Facial Feature Extraction ---
        print("\nExtracting facial features...")
        facial_features_dict = predict_single_image(facial_model_path, pil_img, threshold=0.5, verbose=False)
        print("Facial features:", facial_features_dict)
        # Save facial features to JSON
        save_facial_features_to_json(facial_features_dict, 'facial_features.json')
        # Convert facial features to a string summary for display only
        facial_features_str = ', '.join([f"{k}: {v:.2f}" for k, v in facial_features_dict.items()])

        print(f"\n=== CALLING LLM ===")
        print(f"Sending to LLM: Age={age}, Gender={gender}, Height={height}, Weight={weight}, BMI={bmi:.2f}, Occasion={occasion}")
        print(f"Facial features: {facial_features_str}")
        
        llm_response_text = run_fashion_llm(
            age=age,
            gender=gender,
            height=height,
            weight=weight,
            bmi=bmi,
            occasion=occasion,
            facial_features=facial_features_dict  # ✅ Pass dictionary like manual_test.py
        )

        print(f"\n=== LLM RESPONSE ===")
        print(llm_response_text)
        print("=" * 50)

        if llm_response_text is None or "Error generating recommendations" in llm_response_text:
            print("\nError from LLM, skipping parsing.")
            parsed_recommendations = []
        else:
            parsed_recommendations = parse_llm_recommendations(llm_response_text)

            if parsed_recommendations:
                print("\n=== PARSED RECOMMENDATIONS ===")
                for i, rec in enumerate(parsed_recommendations):
                    print(f"\n--- Recommendation {i+1} ---")
                    for key, value in rec.items():
                        print(f"{key}: {value}")
                
                # Save to JSON for web_scrapping.py
                save_recommendations_to_json(parsed_recommendations, 'llm_recommendations.json')
                print("\nRecommendations saved to llm_recommendations.json.")
                
                # Call web_scrapping.py
                print("\n=== STARTING WEB SCRAPING ===")
                try:
                    result = subprocess.run(["python", "web_scrapping.py"], capture_output=True, text=True)
                    print("Web scraping stdout:", result.stdout)
                    if result.stderr:
                        print("Web scraping stderr:", result.stderr)
                    
                    # Check if scraping output exists
                    if os.path.exists('multi_scraped_output.json'):
                        with open('multi_scraped_output.json', 'r') as f:
                            scraped_data = json.load(f)
                        print(f"\n=== SCRAPING RESULTS ===")
                        print(f"Found {len(scraped_data)} scraped products")
                        
                        for i, product in enumerate(scraped_data[:3]):  # Show first 3 products
                            print(f"\n--- Scraped Product {i+1} ---")
                            print(f"Brand: {product.get('brand', 'N/A')}")
                            print(f"Description: {product.get('description', 'N/A')}")
                            print(f"Price: {product.get('price', 'N/A')}")
                            print(f"Rating: {product.get('rating', 'N/A')}")
                            print(f"Link: {product.get('product_link', 'N/A')[:80]}...")
                        
                        return scraped_data
                    else:
                        print("No scraping output file found!")
                        return []
                        
                except Exception as e:
                    print(f"Error running web scraping: {e}")
                    return []
            else:
                print("\nNo structured recommendations could be parsed from LLM output.")
                print("This might indicate an issue with the LLM response format or parsing logic.")
                return []
                
    except Exception as e:
        print(f"Error processing image: {e}")
        import traceback
        traceback.print_exc()
        return []
