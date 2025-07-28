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
import subprocess

from llm import run_fashion_llm
from extract_info import parse_llm_recommendations, save_recommendations_to_json, save_facial_features_to_json
from ml import predict_single_image
from web_scrapping import scrape_from_llm_recommendations

# Path to the trained facial feature model
facial_model_path = 'celeba_imbalance_aware_classifier.pth'

class AgeGenderModel(nn.Module):
    def __init__(self):  # Fixed: was _init_
        super().__init__()  # Fixed: was _init_
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

try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = AgeGenderModel().to(device)
    model_path = "age_gender_model.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: '{model_path}'.")
    
    # Fixed: Added weights_only=False for PyTorch 2.6+ compatibility
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.eval()
    print("Age/Gender model loaded successfully!")

    scaler_path = 'age_scaler.pkl'
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found: '{scaler_path}'.")
    scaler = joblib.load(scaler_path)
    print("Age scaler loaded successfully!")

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

# Check if webcam is available
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Could not open webcam. Please check if a webcam is connected and accessible.")
    print("You can also test with a static image by modifying this script.")
    exit(1)

print("Press SPACE to capture and run prediction + LLM, ESC to exit.")

# Prompt user for height, weight, and occasion
height = None
weight = None
occasion = None

while height is None:
    try:
        height = float(input("Enter your height in cm: "))
    except ValueError:
        print("Please enter a valid number for height.")

while weight is None:
    try:
        weight = float(input("Enter your weight in kg: "))
    except ValueError:
        print("Please enter a valid number for weight.")

occasion = input("Enter the occasion (e.g., casual, formal, party): ")

print(f"\nConfiguration:")
print(f"Height: {height} cm")
print(f"Weight: {weight} kg") 
print(f"Occasion: {occasion}")
print("\nWebcam window opened. Press SPACE to capture, ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read from webcam")
        break

    cv2.imshow("Capture Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC key
        print("Exiting...")
        break
    elif key == 32:  # SPACE key
        print("\n=== STARTING PREDICTION PIPELINE ===")
        
        # Convert frame to PIL Image
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        input_tensor = transform(pil_img).unsqueeze(0).to(device)

        # Age/Gender prediction
        print("Running age/gender prediction...")
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
        print("\n=== EXTRACTING FACIAL FEATURES ===")
        try:
            facial_features_dict = predict_single_image(facial_model_path, pil_img, threshold=0.3, verbose=False)
            print("Facial features extracted successfully:")
            for feature, value in facial_features_dict.items():
                print(f"  {feature}: {value:.3f}")
            
            # Save facial features to JSON
            save_facial_features_to_json(facial_features_dict, 'facial_features.json')
            print("Facial features saved to facial_features.json")
            
        except Exception as e:
            print(f"Error extracting facial features: {e}")
            facial_features_dict = {}

        # --- LLM Fashion Recommendations ---
        print("\n=== GENERATING FASHION RECOMMENDATIONS ===")
        try:
            llm_response_text = run_fashion_llm(
                age=age,
                gender=gender,
                height=height,
                weight=weight,
                bmi=bmi,
                occasion=occasion,
                facial_features=facial_features_dict
            )
            
            print("LLM Response received!")
            
            if llm_response_text is None:
                print("Error: LLM returned None response")
                parsed_recommendations = []
            else:
                print("Response preview:", llm_response_text[:200] + "..." if len(llm_response_text) > 200 else llm_response_text)
                # Parse recommendations
                parsed_recommendations = parse_llm_recommendations(llm_response_text)

                if parsed_recommendations:
                    print(f"\n=== PARSED RECOMMENDATIONS ({len(parsed_recommendations)} items) ===")
                    for i, rec in enumerate(parsed_recommendations):
                        print(f"\n--- Recommendation {i+1} ---")
                        for key, value in rec.items():
                            print(f"  {key}: {value}")
                    
                    # Save to JSON for web_scrapping.py
                    save_recommendations_to_json(parsed_recommendations, 'llm_recommendations.json')
                    print("\nRecommendations saved to llm_recommendations.json.")
                    
                    # Call web scraping directly with the actual occasion
                    print("\n=== STARTING WEB SCRAPING ===")
                    try:
                        scrape_from_llm_recommendations(
                            'llm_recommendations.json', 
                            'multi_scraped_output.json', 
                            max_products=7, 
                            occasion=occasion  # Pass the actual occasion from user input
                        )
                        print("Web scraping completed successfully!")
                        print("Check multi_scraped_output.json for results.")
                    except Exception as e:
                        print(f"Error running web scraping: {e}")
                        
                else:
                    print("\nNo structured recommendations could be parsed from LLM output.")
                    
        except Exception as e:
            print(f"Error in LLM processing: {e}")

        print("\n=== PIPELINE COMPLETE ===")
        print("Check the following files for outputs:")
        print("- facial_features.json")
        print("- llm_recommendations.json") 
        print("- multi_scraped_output.json")
        break

cap.release()
cv2.destroyAllWindows()
print("Test completed!")
