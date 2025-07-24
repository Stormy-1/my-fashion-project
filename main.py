import os
import cv2
import torch
import timm
import numpy as np
import pandas as pd
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from torchvision import transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load dataset
dataset_path = 'utkface_aligned_cropped\\UTKFace'
age, gender, img_path = [], [], []

print("Loading dataset...")
for file in os.listdir(dataset_path):
    if file.endswith('.jpg'):
        try:
            age_val = int(file.split('_')[0])
            gender_val = int(file.split('_')[1])
            age.append(age_val)
            gender.append(gender_val)
            img_path.append(file)
        except:
            continue

df = pd.DataFrame({'age': age, 'gender': gender, 'img': img_path})
df = df[df['age'] < 80].reset_index(drop=True)
print(f"Dataset loaded: {len(df)} samples")

# Normalize ages
scaler = StandardScaler()
df['age'] = scaler.fit_transform(df[['age']])
df['age'] = df['age'].astype('float32')
df['gender'] = df['gender'].astype('float32')

# Save scaler for later use in prediction
import joblib
joblib.dump(scaler, 'age_scaler.pkl')
print("Age scaler saved")

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")

# Dataset Class
class UTKFaceDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.df = df
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.df.iloc[idx, 2])
        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (200, 200))

        if self.transform:
            image = self.transform(image)

        gender = torch.tensor(self.df.iloc[idx, 1], dtype=torch.float32)
        age = torch.tensor(self.df.iloc[idx, 0], dtype=torch.float32)

        return image, {'gender': gender, 'age': age}

# Transforms
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(200, scale=(0.8, 1.0)),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()
])

# Model with two outputs
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
        gender = torch.sigmoid(self.gender_head(feat)).squeeze(1)
        age = self.age_head(feat).squeeze(1)
        return gender, age

model = AgeGenderModel().to(device)
criterion_gender = nn.BCELoss()
criterion_age = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop with progress bar
def train(train_loader, test_loader):
    print("Starting training...")
    for epoch in range(10):
        model.train()
        running_loss = 0
        running_gender_loss = 0
        running_age_loss = 0
        
        # Training phase with progress bar
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/10 [Train]')
        for imgs, targets in train_pbar:
            imgs = imgs.to(device)
            gender = targets['gender'].to(device)
            age = targets['age'].to(device)

            optimizer.zero_grad()
            gender_pred, age_pred = model(imgs)

            loss_gender = criterion_gender(gender_pred, gender)
            loss_age = criterion_age(age_pred, age)
            loss = 13 * loss_gender + loss_age
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            running_gender_loss += loss_gender.item()
            running_age_loss += loss_age.item()
            
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Gender': f'{loss_gender.item():.4f}',
                'Age': f'{loss_age.item():.4f}'
            })
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_gender_loss = 0
        val_age_loss = 0
        
        with torch.no_grad():
            for imgs, targets in test_loader:
                imgs = imgs.to(device)
                gender = targets['gender'].to(device)
                age = targets['age'].to(device)
                
                gender_pred, age_pred = model(imgs)
                
                loss_gender = criterion_gender(gender_pred, gender)
                loss_age = criterion_age(age_pred, age)
                loss = 13 * loss_gender + loss_age
                
                val_loss += loss.item()
                val_gender_loss += loss_gender.item()
                val_age_loss += loss_age.item()
        
        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(test_loader)
        
        print(f"Epoch [{epoch+1}/10]:")
        print(f"  Train Loss: {avg_train_loss:.4f} (Gender: {running_gender_loss/len(train_loader):.4f}, Age: {running_age_loss/len(train_loader):.4f})")
        print(f"  Val Loss: {avg_val_loss:.4f} (Gender: {val_gender_loss/len(test_loader):.4f}, Age: {val_age_loss/len(test_loader):.4f})")
        
        # Save model checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
        }, f'model_checkpoint_epoch_{epoch+1}.pth')
    
    # Save final model
    torch.save(model.state_dict(), 'age_gender_model.pth')
    print("Training completed! Model saved as 'age_gender_model.pth'")

if __name__ == "__main__":
    
    train_dataset = UTKFaceDataset(train_df, dataset_path, transform=transform)
    test_dataset = UTKFaceDataset(test_df, dataset_path, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    train(train_loader, test_loader)