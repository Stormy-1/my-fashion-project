import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import pandas as pd
from PIL import Image
import os
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import gc
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek

def check_gpu_setup():
    """Check and configure GPU setup"""
    print("=== GPU SETUP CHECK ===")
    
    if torch.cuda.is_available():
        print(f"✓ CUDA is available")
        print(f"✓ CUDA version: {torch.version.cuda}")
        print(f"✓ Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        
        device = torch.device('cuda:0')
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        print(f"✓ Using device: {device}")
        
    else:
        print("⚠ CUDA not available, using CPU")
        device = torch.device('cpu')
    
    print("=" * 25)
    return device

def clear_gpu_cache():
    """Clear GPU cache to free memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

class CelebAImbalanceHandler:
    """Specialized class to handle imbalanced CelebA facial attribute dataset"""
    
    def __init__(self, attr_data, selected_attrs):
        self.attr_data = attr_data
        self.selected_attrs = selected_attrs
        self.class_weights = {}
        self.imbalance_ratios = {}
        
    def analyze_imbalance(self, plot=True):
        """Analyze class imbalance for each attribute"""
        print("=== CELEBA ATTRIBUTE IMBALANCE ANALYSIS ===")
        
        imbalance_info = {}
        
        for attr in self.selected_attrs:
            if attr not in self.attr_data.columns:
                print(f"Warning: Attribute '{attr}' not found in dataset")
                continue
                
            positive_count = (self.attr_data[attr] == 1).sum()
            negative_count = (self.attr_data[attr] == 0).sum()
            total_count = positive_count + negative_count
            
            imbalance_ratio = max(positive_count, negative_count) / min(positive_count, negative_count)
            minority_class = 1 if positive_count < negative_count else 0
            minority_percentage = min(positive_count, negative_count) / total_count * 100
            
            imbalance_info[attr] = {
                'positive_count': positive_count,
                'negative_count': negative_count,
                'total_count': total_count,
                'positive_percentage': positive_count / total_count * 100,
                'negative_percentage': negative_count / total_count * 100,
                'imbalance_ratio': imbalance_ratio,
                'minority_class': minority_class,
                'minority_percentage': minority_percentage
            }
            
            print(f"\n{attr}:")
            print(f"  Positive (1): {positive_count:,} ({positive_count/total_count*100:.1f}%)")
            print(f"  Negative (0): {negative_count:,} ({negative_count/total_count*100:.1f}%)")
            print(f"  Imbalance Ratio: {imbalance_ratio:.2f}:1")
            print(f"  Minority Class: {minority_class} ({minority_percentage:.1f}%)")
            
            if imbalance_ratio > 1.5:
                print(f"  ⚠  IMBALANCED (ratio > 1.5)")
            else:
                print(f"  ✅ BALANCED")
        
        self.imbalance_info = imbalance_info
        
        if plot:
            self._plot_imbalance_analysis()
        
        return imbalance_info
    
    def _plot_imbalance_analysis(self):
        """Plot imbalance analysis"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            attrs = list(self.imbalance_info.keys())
            ratios = [self.imbalance_info[attr]['imbalance_ratio'] for attr in attrs]
            
            # 1. Imbalance ratios
            colors = ['red' if r > 2 else 'orange' if r > 1.5 else 'green' for r in ratios]
            axes[0, 0].barh(attrs, ratios, color=colors)
            axes[0, 0].axvline(x=1.5, color='orange', linestyle='--', label='Threshold (1.5)')
            axes[0, 0].axvline(x=2.0, color='red', linestyle='--', label='Severe (2.0)')
            axes[0, 0].set_xlabel('Imbalance Ratio')
            axes[0, 0].set_title('Attribute Imbalance Ratios')
            axes[0, 0].legend()
            
            # 2. Minority class percentages
            minority_pcts = [self.imbalance_info[attr]['minority_percentage'] for attr in attrs]
            colors2 = ['red' if p < 20 else 'orange' if p < 30 else 'green' for p in minority_pcts]
            axes[0, 1].barh(attrs, minority_pcts, color=colors2)
            axes[0, 1].axvline(x=20, color='red', linestyle='--', label='Severe (<20%)')
            axes[0, 1].axvline(x=30, color='orange', linestyle='--', label='Moderate (<30%)')
            axes[0, 1].set_xlabel('Minority Class Percentage')
            axes[0, 1].set_title('Minority Class Distribution')
            axes[0, 1].legend()
            
            # 3. Positive vs Negative distribution
            pos_pcts = [self.imbalance_info[attr]['positive_percentage'] for attr in attrs]
            neg_pcts = [self.imbalance_info[attr]['negative_percentage'] for attr in attrs]
            
            x = np.arange(len(attrs))
            axes[1, 0].bar(x, pos_pcts, label='Positive (1)', alpha=0.7)
            axes[1, 0].bar(x, neg_pcts, bottom=pos_pcts, label='Negative (0)', alpha=0.7)
            axes[1, 0].set_xticks(x)
            axes[1, 0].set_xticklabels(attrs, rotation=45, ha='right')
            axes[1, 0].set_ylabel('Percentage')
            axes[1, 0].set_title('Positive vs Negative Distribution')
            axes[1, 0].legend()
            
            # 4. Heatmap of imbalance severity
            severity_matrix = []
            for attr in attrs:
                ratio = self.imbalance_info[attr]['imbalance_ratio']
                if ratio > 2:
                    severity = 3
                elif ratio > 1.5:
                    severity = 2
                else:
                    severity = 1
                severity_matrix.append([severity])
            
            im = axes[1, 1].imshow(severity_matrix, cmap='RdYlGn_r', aspect='auto')
            axes[1, 1].set_yticks(range(len(attrs)))
            axes[1, 1].set_yticklabels(attrs)
            axes[1, 1].set_xticks([0])
            axes[1, 1].set_xticklabels(['Imbalance Severity'])
            axes[1, 1].set_title('Imbalance Severity Heatmap')
            
            cbar = plt.colorbar(im, ax=axes[1, 1])
            cbar.set_ticks([1, 2, 3])
            cbar.set_ticklabels(['Balanced', 'Moderate', 'Severe'])
            
            plt.tight_layout()
            plt.savefig('celeba_imbalance_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
        except Exception as e:
            print(f"Could not create plots: {e}")
    
    def compute_class_weights(self, method='balanced'):
        """Compute class weights for each attribute to handle imbalance"""
        print(f"\n=== COMPUTING CLASS WEIGHTS ({method}) ===")
        
        for attr in self.selected_attrs:
            if attr not in self.attr_data.columns:
                continue
                
            y = self.attr_data[attr].values
            
            if method == 'balanced':
                classes = np.unique(y)
                weights = compute_class_weight('balanced', classes=classes, y=y)
                class_weight_dict = dict(zip(classes, weights))
            elif method == 'balanced_subsample':
                classes = np.unique(y)
                weights = compute_class_weight('balanced', classes=classes, y=y)
                weights = np.sqrt(weights)
                class_weight_dict = dict(zip(classes, weights))
            else:
                pos_count = np.sum(y == 1)
                neg_count = np.sum(y == 0)
                total = len(y)
                
                pos_weight = total / (2 * pos_count) if pos_count > 0 else 1
                neg_weight = total / (2 * neg_count) if neg_count > 0 else 1
                
                class_weight_dict = {0: neg_weight, 1: pos_weight}
            
            self.class_weights[attr] = class_weight_dict
            
            print(f"{attr}: Class 0 weight = {class_weight_dict[0]:.3f}, Class 1 weight = {class_weight_dict[1]:.3f}")
        
        return self.class_weights
    
    def create_weighted_sampler(self, indices=None):
        """Create WeightedRandomSampler for balanced batch sampling"""
        print("\n=== CREATING WEIGHTED SAMPLER ===")
        
        if indices is None:
            data_subset = self.attr_data[self.selected_attrs]
        else:
            data_subset = self.attr_data[self.selected_attrs].iloc[indices]
        
        sample_weights = []
        
        for idx, row in data_subset.iterrows():
            total_weight = 0
            for attr in self.selected_attrs:
                if attr in self.class_weights:
                    attr_value = int(row[attr])
                    attr_weight = self.class_weights[attr][attr_value]
                    total_weight += attr_weight
            
            avg_weight = total_weight / len(self.selected_attrs)
            sample_weights.append(avg_weight)
        
        sample_weights = torch.tensor(sample_weights, dtype=torch.float)
        
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        
        print(f"Created weighted sampler for {len(sample_weights)} samples")
        print(f"Weight range: {sample_weights.min():.3f} - {sample_weights.max():.3f}")
        
        return sampler

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance in multi-label classification"""
    
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        p = torch.sigmoid(inputs)
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = p * targets + (1 - p) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class WeightedBCELoss(nn.Module):
    """Weighted Binary Cross Entropy Loss for multi-label classification"""
    
    def __init__(self, class_weights=None):
        super(WeightedBCELoss, self).__init__()
        self.class_weights = class_weights
    
    def forward(self, inputs, targets):
        if self.class_weights is not None:
            weights = torch.ones_like(targets)
            for i, attr_weights in enumerate(self.class_weights.values()):
                weights[:, i] = targets[:, i] * attr_weights[1] + (1 - targets[:, i]) * attr_weights[0]
            
            loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, weight=weights)
        else:
            loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets)
        
        return loss

class CelebADataset(Dataset):
    """Custom Dataset class for CelebA with improved data handling and GPU optimization"""
    
    def __init__(self, img_dir, attr_file, transform=None, selected_attrs=None, pin_memory=True):
        self.img_dir = img_dir
        self.transform = transform
        self.pin_memory = pin_memory
        
        print(f"Loading attributes from: {attr_file}")
        
        try:
            if attr_file.endswith('.csv'):
                self.attr_data = pd.read_csv(attr_file, index_col=0)
            else:
                self.attr_data = pd.read_csv(attr_file, sep='\s+', skiprows=1)
                if not self.attr_data.index.dtype == 'object':
                    self.attr_data.set_index(self.attr_data.columns[0], inplace=True)
        except Exception as e:
            print(f"Error reading attributes file: {e}")
            raise
        
        print(f"Loaded attributes for {len(self.attr_data)} images")
        
        if selected_attrs:
            existing_attrs = [attr for attr in selected_attrs if attr in self.attr_data.columns]
            missing_attrs = [attr for attr in selected_attrs if attr not in self.attr_data.columns]
            
            if missing_attrs:
                print(f"Warning: These attributes not found in dataset: {missing_attrs}")
            
            if not existing_attrs:
                print("No selected attributes found! Using all available attributes.")
                existing_attrs = list(self.attr_data.columns)
            
            self.selected_attrs = existing_attrs
            self.attr_data = self.attr_data[existing_attrs]
        else:
            self.selected_attrs = list(self.attr_data.columns)
        
        print(f"Selected attributes for training: {self.selected_attrs}")
        
        # Convert attribute data to binary format
        print("Converting attribute data to binary format...")
        
        for col in self.attr_data.columns:
            unique_vals = self.attr_data[col].unique()
            self.attr_data[col] = pd.to_numeric(self.attr_data[col], errors='coerce')
            numeric_unique = self.attr_data[col].dropna().unique()
            
            if set(numeric_unique).issubset({-1, 1}):
                self.attr_data[col] = self.attr_data[col].replace(-1, 0)
            elif set(numeric_unique).issubset({0, 1}):
                pass
            else:
                print(f"Warning: Unexpected values in column '{col}': {numeric_unique}")
                self.attr_data[col] = self.attr_data[col].fillna(0)
                self.attr_data[col] = np.clip(self.attr_data[col], 0, 1)
        
        self.attr_data = self.attr_data.fillna(0)
        self.attr_data = self.attr_data.astype(np.float32)
        
        self.img_names = list(self.attr_data.index)
        print(f"Dataset ready with {len(self.img_names)} images and {len(self.selected_attrs)} attributes")
        
    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new('RGB', (64, 64), color='black')
        
        if self.transform:
            image = self.transform(image)
        
        attributes = torch.tensor(self.attr_data.iloc[idx].values, dtype=torch.float32)
        
        if self.pin_memory and torch.cuda.is_available():
            image = image.pin_memory()
            attributes = attributes.pin_memory()
        
        return image, attributes, img_name

class FacialFeatureClassifier(nn.Module):
    """CNN Model for Multi-label Classification"""
    
    def __init__(self, num_attributes):
        super(FacialFeatureClassifier, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_attributes),
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# ==================== PREDICTION FUNCTIONALITY ====================

import sys
import io
from contextlib import redirect_stdout, redirect_stderr

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

def apply_custom_facial_feature_threshold(probabilities, threshold=0.3):
    """
    Apply custom thresholding logic for facial features:
    - For single features: use threshold (default 0.3)
    - For categorical features: select only the highest value in each category
    
    Args:
        probabilities (dict): Raw probabilities from model
        threshold (float): Threshold for single features (default: 0.3)
        
    Returns:
        dict: Processed probabilities with custom thresholding applied
    """
    # Define categorical feature groups
    hair_color_features = ['Black_Hair', 'Blond_Hair', 'Brown_Hair']
    eyebrow_features = ['Bushy_Eyebrows', 'Arched_Eyebrows']
    hair_texture_features = ['Straight_Hair', 'Wavy_Hair']
    
    # Group all categorical features
    categorical_groups = [
        hair_color_features,
        eyebrow_features, 
        hair_texture_features
    ]
    
    # Get all categorical feature names
    all_categorical_features = set()
    for group in categorical_groups:
        all_categorical_features.update(group)
    
    # Process probabilities
    processed_probabilities = {}
    
    # Process categorical features (select highest in each group)
    for group in categorical_groups:
        group_probs = {}
        for feature in group:
            if feature in probabilities:
                group_probs[feature] = probabilities[feature]
        
        if group_probs:
            # Find the feature with highest probability in this group
            max_feature = max(group_probs, key=group_probs.get)
            max_prob = group_probs[max_feature]
            
            # Set the highest feature to its probability, others to 0
            for feature in group:
                if feature in probabilities:
                    if feature == max_feature:
                        processed_probabilities[feature] = max_prob
                    else:
                        processed_probabilities[feature] = 0.0
    
    # Process single features (apply threshold)
    for feature, prob in probabilities.items():
        if feature not in all_categorical_features:
            # Apply threshold: above threshold = keep probability, below = 0
            processed_probabilities[feature] = prob if prob > threshold else 0.0
    
    return processed_probabilities

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

def predict_single_image(model_path, image_path, threshold=0.3, verbose=False):
    """
    Quick function to predict on single image without printing status messages
    
    Args:
        model_path (str): Path to trained model
        image_path (str): Path to image or PIL Image
        threshold (float): Classification threshold for single features (default: 0.3)
        verbose (bool): Whether to print status messages (default: False)
        
    Returns:
        dict: Processed probabilities of selected facial attributes with custom thresholding
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
        results = predictor.predict(image_path, threshold=0.5)  # Use 0.5 for raw prediction
    else:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            predictor = FacialFeaturePredictor(model_path)
            results = predictor.predict(image_path, threshold=0.5)  # Use 0.5 for raw prediction

    # Extract probabilities only for selected attributes
    probabilities = results['probabilities']
    selected_probabilities = {}
    
    for attr in selected_attrs:
        if attr in probabilities:
            selected_probabilities[attr] = float(probabilities[attr])
    
    # Apply custom thresholding logic
    processed_probabilities = apply_custom_facial_feature_threshold(selected_probabilities, threshold)
            
    return processed_probabilities
