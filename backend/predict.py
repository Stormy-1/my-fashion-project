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
    def _init_(self):
        super()._init_()
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

    model = AgeGenderModel().to(device)
    model_path = "age_gender_model.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: '{model_path}'.")
    model.load_state_dict(torch.load(model_path, map_location=device))
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

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Could not open webcam. Please check if a webcam is connected and accessible.")
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


while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Capture Frame", frame)
    key = cv2.waitKey(1)

    if key == 27:
        break
    elif key == 32:
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
        # Convert facial features to a string summary for LLM
        facial_features_str = ', '.join([f"{k}: {v:.2f}" for k, v in facial_features_dict.items()])

        llm_response_text = run_fashion_llm(
            age=age,
            gender=gender,
            height=height,
            weight=weight,
            bmi=bmi,
            occasion=occasion,
            facial_features=facial_features_str
        )


        if "Error generating recommendations" in llm_response_text:
            print("\nError from LLM, skipping parsing.")
            parsed_recommendations = []
        else:
            parsed_recommendations = parse_llm_recommendations(llm_response_text)

            if parsed_recommendations:
                print("\n--- Parsed Recommendations ---")
                for i, rec in enumerate(parsed_recommendations):
                    print(f"\n--- Recommendation {i+1} ---")
                    for key, value in rec.items():
                        print(f"{key}: {value}")
                # Save to JSON for web_scrapping.py
                save_recommendations_to_json(parsed_recommendations, 'llm_recommendations.json')
                print("\nRecommendations saved to llm_recommendations.json.")
                # Call web_scrapping.py
                print("\nStarting web scraping for recommendations...")
                subprocess.run(["python", "web_scrapping.py"])
            else:
                print("\nNo structured recommendations could be parsed from LLM output.")

        break

cap.release()
cv2.destroyAllWindows()