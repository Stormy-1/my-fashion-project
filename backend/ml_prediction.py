# ==================== PREDICTION FUNCTIONALITY ====================

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import sys
import io
from contextlib import redirect_stdout, redirect_stderr

# Import the model class from ml_new.py
from ml_new import FacialFeatureClassifier, check_gpu_setup

class FacialFeaturePredictor:
    """Class for making predictions on single images with trained CelebA model"""
    
    def __init__(self, model_path, device=None):
        """
        Initialize predictor with trained model
        
        Args:
            model_path (str): Path to saved model file (.pth)
            device (torch.device): Device to run inference on
        """
        self.device = device if device else check_gpu_setup()
        self.model = None
        self.selected_attrs = None
        self.transform = None
        self.imbalance_info = None
        self.class_weights = None
        
        self.load_model(model_path)
        self.setup_transforms()
        
    def load_model(self, model_path):
        """Load trained model from checkpoint"""
        print(f"Loading model from: {model_path}")
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # Extract model configuration
            self.selected_attrs = checkpoint['selected_attrs']
            model_config = checkpoint['model_config']
            num_attributes = model_config['num_attributes']
            
            # Load additional info if available
            if 'imbalance_info' in checkpoint:
                self.imbalance_info = checkpoint['imbalance_info']
            if 'class_weights' in checkpoint:
                self.class_weights = checkpoint['class_weights']
            
            # Initialize model
            self.model = FacialFeatureClassifier(num_attributes=num_attributes)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            print(f"✓ Model loaded successfully")
            print(f"✓ Attributes: {len(self.selected_attrs)}")
            print(f"✓ Selected attributes: {self.selected_attrs}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def setup_transforms(self):
        """Setup image preprocessing transforms"""
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        print("✓ Image transforms configured")
    
    def preprocess_image(self, image_input):
        """
        Preprocess image for inference
        
        Args:
            image_input: PIL Image, numpy array, or path to image file
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        try:
            # Handle different input types
            if isinstance(image_input, str):
                # Path to image file
                image = Image.open(image_input).convert('RGB')
            elif isinstance(image_input, np.ndarray):
                # Numpy array
                image = Image.fromarray(image_input.astype('uint8')).convert('RGB')
            elif hasattr(image_input, 'convert'):
                # PIL Image
                image = image_input.convert('RGB')
            else:
                raise ValueError("Unsupported image input type")
            
            # Apply transforms
            if self.transform:
                image_tensor = self.transform(image)
            else:
                # Fallback transform
                image_tensor = transforms.ToTensor()(image)
            
            # Add batch dimension
            image_tensor = image_tensor.unsqueeze(0)
            
            return image_tensor
            
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            raise
    
    def predict(self, image_input, threshold=0.5, return_probabilities=False):
        """
        Make prediction on single image
        
        Args:
            image_input: Image input (path, PIL Image, or numpy array)
            threshold (float): Classification threshold (default: 0.5)
            return_probabilities (bool): Whether to return raw probabilities
            
        Returns:
            dict: Prediction results with attributes and probabilities/predictions
        """
        print("Making prediction...")
        
        try:
            # Preprocess image
            image_tensor = self.preprocess_image(image_input)
            image_tensor = image_tensor.to(self.device, non_blocking=True)
            
            # Make prediction
            with torch.no_grad():
                if self.device.type == 'cuda':
                    with torch.amp.autocast('cuda'):
                        outputs = self.model(image_tensor)
                else:
                    outputs = self.model(image_tensor)
                
                # Apply sigmoid to get probabilities
                probabilities = torch.sigmoid(outputs).squeeze(0)  # Remove batch dimension
                
                # Apply threshold for binary predictions
                predictions = (probabilities > threshold).float()
            
            # Convert to numpy for easier handling
            probabilities_np = probabilities.cpu().numpy()
            predictions_np = predictions.cpu().numpy()
            
            # Create results dictionary
            results = {
                'attributes': self.selected_attrs,
                'probabilities': dict(zip(self.selected_attrs, probabilities_np)),
                'predictions': dict(zip(self.selected_attrs, predictions_np.astype(int))),
                'threshold': threshold
            }
            
            if return_probabilities:
                results['raw_probabilities'] = probabilities_np
                results['raw_predictions'] = predictions_np
            
            print("✓ Prediction completed successfully")
            return results
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            raise
    
    def predict_with_confidence(self, image_input, confidence_threshold=0.7):
        """
        Make prediction with confidence filtering
        
        Args:
            image_input: Image input
            confidence_threshold (float): Minimum confidence for positive prediction
            
        Returns:
            dict: Filtered predictions with confidence scores
        """
        results = self.predict(image_input, return_probabilities=True)
        
        confident_positive = {}
        confident_negative = {}
        uncertain = {}
        
        for attr, prob in results['probabilities'].items():
            if prob >= confidence_threshold:
                confident_positive[attr] = prob
            elif prob <= (1 - confidence_threshold):
                confident_negative[attr] = prob
            else:
                uncertain[attr] = prob
        
        return {
            'confident_positive': confident_positive,
            'confident_negative': confident_negative,
            'uncertain': uncertain,
            'all_probabilities': results['probabilities']
        }
    
    def visualize_predictions(self, image_input, save_path=None, figsize=(15, 10)):
        """
        Visualize image with prediction results
        
        Args:
            image_input: Image input
            save_path (str): Path to save visualization
            figsize (tuple): Figure size
        """
        try:
            # Make prediction
            results = self.predict(image_input, return_probabilities=True)
            
            # Load and display original image
            if isinstance(image_input, str):
                original_image = Image.open(image_input).convert('RGB')
            elif isinstance(image_input, np.ndarray):
                original_image = Image.fromarray(image_input.astype('uint8')).convert('RGB')
            else:
                original_image = image_input.convert('RGB')
            
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
            
            # Display original image
            ax1.imshow(original_image)
            ax1.set_title('Input Image', fontsize=14, fontweight='bold')
            ax1.axis('off')
            
            # Display prediction probabilities
            attrs = list(results['probabilities'].keys())
            probs = list(results['probabilities'].values())
            preds = [results['predictions'][attr] for attr in attrs]
            
            # Color code based on predictions
            colors = ['green' if pred == 1 else 'red' for pred in preds]
            
            y_pos = np.arange(len(attrs))
            bars = ax2.barh(y_pos, probs, color=colors, alpha=0.7)
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(attrs)
            ax2.set_xlabel('Probability')
            ax2.set_title('Attribute Probabilities', fontsize=14, fontweight='bold')
            ax2.axvline(x=0.5, color='black', linestyle='--', alpha=0.5)
            ax2.set_xlim(0, 1)
            
            # Add probability values on bars
            for i, (bar, prob) in enumerate(zip(bars, probs)):
                ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                        f'{prob:.3f}', va='center', fontsize=8)
            
            # Create summary text
            positive_attrs = [attr for attr, pred in results['predictions'].items() if pred == 1]
            negative_attrs = [attr for attr, pred in results['predictions'].items() if pred == 0]
            
            summary_text = f"""
PREDICTION SUMMARY

✓ POSITIVE ATTRIBUTES ({len(positive_attrs)}):
{', '.join(positive_attrs) if positive_attrs else 'None'}

✗ NEGATIVE ATTRIBUTES ({len(negative_attrs)}):
{', '.join(negative_attrs) if negative_attrs else 'None'}

CONFIDENCE ANALYSIS:
High Confidence (>0.8): {len([p for p in probs if p > 0.8 or p < 0.2])}
Medium Confidence (0.6-0.8): {len([p for p in probs if 0.2 <= p <= 0.4 or 0.6 <= p <= 0.8])}
Low Confidence (0.4-0.6): {len([p for p in probs if 0.4 < p < 0.6])}
            """
            
            ax3.text(0.05, 0.95, summary_text, transform=ax3.transAxes, 
                    fontsize=10, verticalalignment='top', fontfamily='monospace')
            ax3.set_title('Prediction Summary', fontsize=14, fontweight='bold')
            ax3.axis('off')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"✓ Visualization saved to: {save_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"Error creating visualization: {e}")
            raise

# Standalone prediction functions
def load_trained_model(model_path, device=None):
    """
    Load trained model for prediction
    
    Args:
        model_path (str): Path to saved model
        device: Device to load model on
        
    Returns:
        FacialFeaturePredictor: Initialized predictor
    """
    return FacialFeaturePredictor(model_path, device)

def predict_single_image(model_path, image_path, threshold=0.5, verbose=False):
    """
    Quick function to predict on single image without printing status messages
    
    Args:
        model_path (str): Path to trained model
        image_path (str): Path to image
        threshold (float): Classification threshold (default: 0.5)
        verbose (bool): Whether to print status messages (default: False)
        
    Returns:
        dict: Probabilities of selected facial attributes
    """
    # Define the specific attributes to return (in order)
    selected_attrs = [
        'Straight_Hair', 'Wavy_Hair', 'No_Beard', 'Oval_Face', 'Pale_Skin',
        'Pointy_Nose', 'Receding_Hairline', 'Mustache', 'Big_Lips', 'Big_Nose',
        'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Bushy_Eyebrows', 
        'Arched_Eyebrows', 'Bald'
    ]

    if verbose:
        predictor = FacialFeaturePredictor(model_path)
        results = predictor.predict(image_path, threshold=threshold)
    else:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            predictor = FacialFeaturePredictor(model_path)
            results = predictor.predict(image_path, threshold=threshold)

    # Extract probabilities only for selected attributes
    probabilities = results['probabilities']
    selected_probabilities = {}
    
    for attr in selected_attrs:
        if attr in probabilities:
            selected_probabilities[attr] = float(probabilities[attr])
            
    return selected_probabilities
