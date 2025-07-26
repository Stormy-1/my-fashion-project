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
    
    def _init_(self, attr_data, selected_attrs):
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
    
    def _init_(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self)._init_()
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
    
    def _init_(self, class_weights=None):
        super(WeightedBCELoss, self)._init_()
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
    
    def _init_(self, img_dir, attr_file, transform=None, selected_attrs=None, pin_memory=True):
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
        
    def _len_(self):
        return len(self.img_names)
    
    def _getitem_(self, idx):
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
    
    def _init_(self, num_attributes):
        super(FacialFeatureClassifier, self)._init_()
        
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

class FacialFeaturePredictor:
    """Class for making predictions on single images with trained CelebA model"""
    
    def _init_(self, model_path, device=None):
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
            checkpoint = torch.load(model_path, map_location=self.device)
            
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


import sys
import io
from contextlib import redirect_stdout, redirect_stderr

def predict_single_image(model_path, image_path, threshold=0.5, verbose=False):
    """
    Quick function to predict on single image without printing status messages
    
    Args:
        model_path (str): Path to trained model
        image_path (str): Path to image
        threshold (float): Classification threshold (default: 0.5)
        verbose (bool): Whether to print status messages (default: False)
        
    Returns:
        list: Probabilities of selected facial attributes in the specified order
    """
    import io
    import sys
    from contextlib import redirect_stdout, redirect_stderr

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
            selected_probabilities[attr]=float(probabilities[attr])
            
    return selected_probabilities

# ==================== TRAINING FUNCTIONS (ORIGINAL) ====================

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10, device='cuda'):
    """Training function with imbalance-aware monitoring"""
    train_losses = []
    val_losses = []
    val_accuracies = []
    val_f1_scores = []
    
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    
    for epoch in range(num_epochs):
        clear_gpu_cache()
        
        # Training phase
        model.train()
        running_loss = 0.0
        
        for i, (images, labels, _) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Training')):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item()
            
            if i % 50 == 0 and device.type == 'cuda':
                torch.cuda.empty_cache()
        
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                
                if scaler is not None:
                    with torch.amp.autocast('cuda'):
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                
                probabilities = torch.sigmoid(outputs)
                preds = (probabilities > 0.5).float()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Calculate metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        accuracy = accuracy_score(all_labels.flatten(), all_preds.flatten())
        f1 = f1_score(all_labels.flatten(), all_preds.flatten(), average='weighted')
        
        val_accuracies.append(accuracy)
        val_f1_scores.append(f1)
        
        scheduler.step()
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        print(f'Val Acc: {accuracy:.4f}, Val F1: {f1:.4f}')
        print(f'Learning Rate: {scheduler.get_last_lr()[0]:.6f}')
        
        if device.type == 'cuda':
            memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(device) / 1024**3
            print(f'GPU Memory - Allocated: {memory_allocated:.2f} GB, Reserved: {memory_reserved:.2f} GB')
        
        print('-' * 60)
    
    return train_losses, val_losses, val_accuracies, val_f1_scores

def evaluate_model_enhanced(model, test_loader, selected_attrs, device='cuda'):
    """Enhanced evaluation function with detailed metrics for imbalanced data"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels, _ in tqdm(test_loader, desc='Evaluating'):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            if device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
            else:
                outputs = model(images)
            
            probabilities = torch.sigmoid(outputs)
            preds = (probabilities > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate comprehensive metrics
    print("\n=== DETAILED EVALUATION RESULTS ===")
    
    attr_metrics = {}
    for i, attr in enumerate(selected_attrs):
        attr_acc = accuracy_score(all_labels[:, i], all_preds[:, i])
        attr_f1 = f1_score(all_labels[:, i], all_preds[:, i], average='binary')
        
        attr_metrics[attr] = {
            'accuracy': attr_acc,
            'f1_score': attr_f1
        }
        
        print(f'{attr}:')
        print(f'  Accuracy: {attr_acc:.4f}')
        print(f'  F1-Score: {attr_f1:.4f}')
    
    # Overall metrics
    overall_acc = accuracy_score(all_labels.flatten(), all_preds.flatten())
    overall_f1 = f1_score(all_labels.flatten(), all_preds.flatten(), average='weighted')
    
    print(f'\nOverall Accuracy: {overall_acc:.4f}')
    print(f'Overall F1-Score: {overall_f1:.4f}')
    
    return attr_metrics, overall_acc, overall_f1

def get_improved_loss_function(imbalance_handler, loss_type='focal'):
    """Get improved loss function for imbalanced data"""
    print(f"\n=== CONFIGURING {loss_type.upper()} LOSS ===")
    
    if loss_type == 'focal':
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
        print("Using Focal Loss (alpha=0.25, gamma=2.0)")
    elif loss_type == 'weighted_bce':
        criterion = WeightedBCELoss(class_weights=imbalance_handler.class_weights)
        print("Using Weighted BCE Loss with computed class weights")
    else:
        criterion = nn.BCEWithLogitsLoss()
        print("Using standard BCEWithLogitsLoss")
    
    return criterion

def print_balancing_recommendations(imbalance_info):
    """Print recommendations for handling imbalance"""
    print("\n=== IMBALANCE HANDLING RECOMMENDATIONS ===")
    
    severe_imbalance = []
    moderate_imbalance = []
    balanced_attrs = []
    
    for attr, info in imbalance_info.items():
        ratio = info['imbalance_ratio']
        if ratio > 2:
            severe_imbalance.append(attr)
        elif ratio > 1.5:
            moderate_imbalance.append(attr)
        else:
            balanced_attrs.append(attr)
    
    print(f"✅ Balanced attributes ({len(balanced_attrs)}): {balanced_attrs}")
    print(f"⚠  Moderately imbalanced ({len(moderate_imbalance)}): {moderate_imbalance}")
    print(f"🚨 Severely imbalanced ({len(severe_imbalance)}): {severe_imbalance}")
    
    print("\n📋 RECOMMENDED STRATEGIES:")
    
    if severe_imbalance:
        print("For severely imbalanced attributes:")
        print("  1. Use Focal Loss (alpha=0.25, gamma=2.0)")
        print("  2. Apply SMOTE for oversampling")
        print("  3. Use WeightedRandomSampler")
        print("  4. Consider ensemble methods")
    
    if moderate_imbalance:
        print("For moderately imbalanced attributes:")
        print("  1. Use class weights in loss function")
        print("  2. Apply data augmentation")
        print("  3. Use stratified sampling")
    
    if balanced_attrs:
        print("For balanced attributes:")
        print("  1. Standard training approaches work well")
        print("  2. Focus on model architecture and regularization")

def integrate_imbalance_handling(celeba_dataset, selected_attrs):
    """Main function to integrate imbalance handling with CelebA code"""
    print("=== INTEGRATING IMBALANCE HANDLING ===")
    
    attr_data = celeba_dataset.attr_data
    handler = CelebAImbalanceHandler(attr_data, selected_attrs)
    
    # Analyze imbalance
    imbalance_info = handler.analyze_imbalance(plot=True)
    
    # Compute class weights
    class_weights = handler.compute_class_weights(method='balanced')
    
    # Print recommendations
    print_balancing_recommendations(imbalance_info)
    
    return {
        'handler': handler,
        'imbalance_info': imbalance_info,
        'class_weights': class_weights,
        'focal_loss': get_improved_loss_function(handler, 'focal'),
        'weighted_bce_loss': get_improved_loss_function(handler, 'weighted_bce'),
        'recommendations': {
            'use_focal_loss': any(info['imbalance_ratio'] > 2 for info in imbalance_info.values()),
            'use_weighted_sampler': any(info['imbalance_ratio'] > 1.5 for info in imbalance_info.values()),
            'apply_smote': any(info['minority_percentage'] < 20 for info in imbalance_info.values())
        }
    }

def main_with_imbalance_handling():
    """Complete main function with integrated imbalance handling"""
    
    # Setup
    device = check_gpu_setup()
    
    # Configuration - UPDATE THESE PATHS
    IMG_DIR = 'img_align_celeba'
    ATTR_FILE = 'list_attr_celeba.csv'
    
    # Selected facial features
    SELECTED_ATTRS = [
        'Attractive', 'Arched_Eyebrows', 'Bald', 'Big_Lips', 'Big_Nose', 
        'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Bushy_Eyebrows', 'Eyeglasses',
        'High_Cheekbones', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face',
        'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 
        'Straight_Hair', 'Wavy_Hair'
    ]
    
    print(f"Target attributes for classification: {SELECTED_ATTRS}")
    print(f"Number of attributes: {len(SELECTED_ATTRS)}")
    
    # Data transforms
    transform_train = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_val = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset
    print("Creating dataset...")
    full_dataset = CelebADataset(IMG_DIR, ATTR_FILE, transform=transform_train, 
                               selected_attrs=SELECTED_ATTRS, pin_memory=(device.type == 'cuda'))
    
    # Update selected attributes to only include found ones
    SELECTED_ATTRS = full_dataset.selected_attrs
    print(f"Final selected attributes: {SELECTED_ATTRS}")
    
    # === IMBALANCE HANDLING INTEGRATION ===
    print("\n" + "="*50)
    print("ANALYZING AND HANDLING DATASET IMBALANCE")
    print("="*50)
    
    # Integrate imbalance handling
    imbalance_components = integrate_imbalance_handling(full_dataset, SELECTED_ATTRS)
    
    # Split dataset
    dataset_size = len(full_dataset)
    train_size = int(0.7 * dataset_size)
    val_size = int(0.15 * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    # Get indices for splits
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
    
    # Apply different transforms to validation and test sets
    val_dataset.dataset.transform = transform_val
    test_dataset.dataset.transform = transform_val
    
    # Create data loaders with imbalance handling
    BATCH_SIZE = 64 if device.type == 'cuda' else 8
    NUM_WORKERS = 0 if os.name == 'nt' else (4 if device.type == 'cuda' else 2)
    
    # Create weighted sampler for training if recommended
    if imbalance_components['recommendations']['use_weighted_sampler']:
        print("\n🎯 Using WeightedRandomSampler for balanced training batches")
        train_sampler = imbalance_components['handler'].create_weighted_sampler(train_indices)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler,
                                num_workers=NUM_WORKERS, pin_memory=(device.type == 'cuda'))
    else:
        print("\n📊 Using standard random sampling")
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                num_workers=NUM_WORKERS, pin_memory=(device.type == 'cuda'))
    
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=(device.type == 'cuda'))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=NUM_WORKERS, pin_memory=(device.type == 'cuda'))
    
    # Initialize model
    model = FacialFeatureClassifier(num_attributes=len(SELECTED_ATTRS))
    model = model.to(device)
    
    # Use DataParallel if multiple GPUs available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)
    
    # Use improved loss function based on imbalance analysis
    if imbalance_components['recommendations']['use_focal_loss']:
        print("\n🎯 Using Focal Loss for severe imbalance")
        criterion = imbalance_components['focal_loss']
    else:
        print("\n📊 Using Weighted BCE Loss")
        criterion = imbalance_components['weighted_bce_loss']
    
    # Optimizer with adjusted learning rate for imbalanced data
    if any(info['imbalance_ratio'] > 3 for info in imbalance_components['imbalance_info'].values()):
        LEARNING_RATE = 0.0005
        print(f"🎯 Using reduced learning rate: {LEARNING_RATE}")
    else:
        LEARNING_RATE = 0.001
        print(f"📊 Using standard learning rate: {LEARNING_RATE}")
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.8)
    
    # Training
    NUM_EPOCHS = 15
    print(f'\n🚀 Starting training with imbalance-aware configuration...')
    print(f'📊 Model: {sum(p.numel() for p in model.parameters())} parameters')
    print(f'🎯 Loss: {type(criterion)._name_}')
    print(f'📈 Epochs: {NUM_EPOCHS}')
    
    # Train the model
    train_losses, val_losses, val_accuracies, val_f1_scores = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, NUM_EPOCHS, device
    )
    
    # Enhanced evaluation for imbalanced data
    print('\n=== IMBALANCE-AWARE EVALUATION ===')
    attr_metrics, overall_acc, overall_f1 = evaluate_model_enhanced(model, test_loader, SELECTED_ATTRS, device)
    
    # Save the model
    model_save_path = 'celeba_imbalance_aware_classifier.pth'
    model_to_save = model.module if hasattr(model, 'module') else model
    torch.save({
        'model_state_dict': model_to_save.state_dict(),
        'selected_attrs': SELECTED_ATTRS,
        'imbalance_info': imbalance_components['imbalance_info'],
        'class_weights': imbalance_components['class_weights'],
        'model_config': {
            'num_attributes': len(SELECTED_ATTRS),
            'image_size': 64
        }
    }, model_save_path)
    
    print(f'\nModel saved as {model_save_path}')
    
    # Print final summary
    print(f'\n=== TRAINING SUMMARY ===')
    print(f'Successfully trained imbalance-aware classifier for {len(SELECTED_ATTRS)} facial features')
    print(f'Overall Test Accuracy: {overall_acc:.4f}')
    print(f'Overall Test F1-Score: {overall_f1:.4f}')
    
    # Clear GPU cache
    clear_gpu_cache()
    
    # Plot comprehensive results
    try:
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # Training curves
        axes[0, 0].plot(train_losses, label='Train Loss', color='blue')
        axes[0, 0].plot(val_losses, label='Val Loss', color='red')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        axes[0, 1].plot(val_accuracies, label='Accuracy', color='green')
        axes[0, 1].plot(val_f1_scores, label='F1-Score', color='orange')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].set_title('Validation Metrics')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Per-attribute results
        attrs = list(attr_metrics.keys())
        accuracies = [attr_metrics[attr]['accuracy'] for attr in attrs]
        f1_scores = [attr_metrics[attr]['f1_score'] for attr in attrs]
        
        y_pos = np.arange(len(attrs))
        axes[0, 2].barh(y_pos, accuracies, alpha=0.7, label='Accuracy')
        axes[0, 2].barh(y_pos, f1_scores, alpha=0.7, label='F1-Score')
        axes[0, 2].set_yticks(y_pos)
        axes[0, 2].set_yticklabels(attrs)
        axes[0, 2].set_xlabel('Score')
        axes[0, 2].set_title('Per-Attribute Performance')
        axes[0, 2].legend()
        
        # Imbalance analysis
        ratios = [imbalance_components['imbalance_info'][attr]['imbalance_ratio'] for attr in attrs]
        colors = ['red' if r > 2 else 'orange' if r > 1.5 else 'green' for r in ratios]
        
        axes[1, 0].barh(attrs, ratios, color=colors)
        axes[1, 0].axvline(x=1.5, color='orange', linestyle='--', label='Moderate (1.5)')
        axes[1, 0].axvline(x=2.0, color='red', linestyle='--', label='Severe (2.0)')
        axes[1, 0].set_xlabel('Imbalance Ratio')
        axes[1, 0].set_title('Dataset Imbalance Analysis')
        axes[1, 0].legend()
        
        # Performance vs Imbalance correlation
        axes[1, 1].scatter(ratios, accuracies, alpha=0.7, c=colors)
        axes[1, 1].set_xlabel('Imbalance Ratio')
        axes[1, 1].set_ylabel('Test Accuracy')
        axes[1, 1].set_title('Performance vs Imbalance')
        axes[1, 1].grid(True)
        
        # Summary statistics
        axes[1, 2].axis('off')
        summary_text = f"""
        IMBALANCE-AWARE CLASSIFICATION RESULTS
        
        Total Attributes: {len(SELECTED_ATTRS)}
        
        Balanced Attributes: {len([attr for attr, info in imbalance_components['imbalance_info'].items() if info['imbalance_ratio'] <= 1.5])}
        Moderately Imbalanced: {len([attr for attr, info in imbalance_components['imbalance_info'].items() if 1.5 < info['imbalance_ratio'] <= 2])}
        Severely Imbalanced: {len([attr for attr, info in imbalance_components['imbalance_info'].items() if info['imbalance_ratio'] > 2])}
        
        Overall Test Accuracy: {overall_acc:.4f}
        Overall Test F1-Score: {overall_f1:.4f}
        
        Loss Function: {type(criterion)._name_}
        Weighted Sampling: {imbalance_components['recommendations']['use_weighted_sampler']}
        """
        axes[1, 2].text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center')
        
        plt.tight_layout()
        plt.savefig('imbalance_aware_facial_classification_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Comprehensive results saved as 'imbalance_aware_facial_classification_results.png'")
        
    except Exception as e:
        print(f"Could not display plots: {e}")

# ==================== USAGE EXAMPLES ====================

def demo_prediction_usage():
    """Demo showing how to use the prediction functionality"""
    print("\n" + "="*60)
    print("PREDICTION USAGE EXAMPLES")
    print("="*60)
    
    # Example 1: Basic prediction
    print("\n1. Basic Single Image Prediction:")
    print("="*40)
    print("""
# Load trained model and predict
predictor = FacialFeaturePredictor('celeba_imbalance_aware_classifier.pth')
results = predictor.predict('path/to/image.jpg')

# Print results
print("Predicted attributes:")
for attr, pred in results['predictions'].items():
    if pred == 1:
        prob = results['probabilities'][attr]
        print(f"  ✓ {attr}: {prob:.3f}")
    """)
    
    # Example 2: Confidence-based prediction
    print("\n2. Confidence-based Prediction:")
    print("="*40)
    print("""
# Get predictions with confidence filtering
confident_results = predictor.predict_with_confidence('image.jpg', confidence_threshold=0.8)

print("High confidence positive attributes:")
for attr, prob in confident_results['confident_positive'].items():
    print(f"  ✓ {attr}: {prob:.3f}")

print("Uncertain attributes:")
for attr, prob in confident_results['uncertain'].items():
    print(f"  ? {attr}: {prob:.3f}")
    """)
    
    # Example 3: Visualization
    print("\n3. Prediction with Visualization:")
    print("="*40)
    print("""
# Visualize predictions
predictor.visualize_predictions('image.jpg', save_path='prediction_result.png')
    """)
    
    # Example 4: Batch prediction
    print("\n4. Batch Prediction:")
    print("="*40)
    print("""
# Predict on multiple images
image_paths = ['img1.jpg', 'img2.jpg', 'img3.jpg']
batch_results = batch_predict_images('model.pth', image_paths, save_results=True)
    """)
    
    # Example 5: Quick prediction function
    print("\n5. Quick Prediction Function:")
    print("="*40)
    print("""
# One-line prediction with visualization
results = predict_single_image('model.pth', 'image.jpg', threshold=0.6, visualize=True)
    """)

if __name__ == "__main__":
    main_with_imbalance_handling()