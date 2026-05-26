import os
import sys
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Ensure the project root is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models_def import SpeechEmotionModel
from utils import MultimodalDataset

def train():
    print("=== Training Speech-only Emotion Recognition Model ===")
    
    # 1. Load cached data
    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cache_path = os.path.join(project_dir, "cached_data.pkl")
    
    if not os.path.exists(cache_path):
        print(f"Error: Cache file {cache_path} not found. Please run cache_data.py first.")
        return
        
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
        
    train_samples = cache["train_samples"]
    val_samples = cache["val_samples"]
    
    # 2. Datasets & Loaders
    train_dataset = MultimodalDataset(train_samples)
    val_dataset = MultimodalDataset(val_samples)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # 3. Model, Loss, Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Using device: {device}")
    
    model = SpeechEmotionModel(input_dim=40, hidden_dim=64, num_layers=2, num_classes=7).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # 4. Training Loop
    epochs = 40
    best_val_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    
    pipeline_dir = os.path.dirname(os.path.abspath(__file__))
    model_save_path = os.path.join(pipeline_dir, "best_speech_model.pth")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch in train_loader:
            speech = batch["speech"].to(device) # [batch, time_steps, 40]
            labels = batch["label"].to(device)
            
            optimizer.zero_grad()
            outputs = model(speech)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * speech.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += speech.size(0)
            
        train_loss /= total_train
        train_acc = correct_train / total_train
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for batch in val_loader:
                speech = batch["speech"].to(device)
                labels = batch["label"].to(device)
                
                outputs = model(speech)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * speech.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += (preds == labels).sum().item()
                total_val += speech.size(0)
                
        val_loss /= total_val
        val_acc = correct_val / total_val
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"  --> Saved new best speech model with Val Acc: {val_acc*100:.2f}%")
            
    # Save training history
    history_save_path = os.path.join(pipeline_dir, "speech_history.pkl")
    with open(history_save_path, "wb") as f:
        pickle.dump(history, f)
        
    print(f"\n✅ Speech training completed. Best Val Acc: {best_val_acc*100:.2f}%")
    print(f"Model saved to: {model_save_path}")

if __name__ == "__main__":
    train()
