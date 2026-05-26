import os
import sys
import pickle
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Ensure project root is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from project.models_def import SpeechEmotionModel, TextEmotionModel, MultimodalFusionModel
from project.utils import MultimodalDataset, EMOTIONS

def test():
    print("=== Testing Multimodal Fusion Emotion Recognition Model ===")
    
    # 1. Load cached data
    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cache_path = os.path.join(project_dir, "cached_data.pkl")
    
    if not os.path.exists(cache_path):
        print(f"Error: Cache file {cache_path} not found.")
        return
        
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
        
    test_samples = cache["test_samples"]
    word_to_idx = cache["word_to_idx"]
    
    test_dataset = MultimodalDataset(test_samples)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # 2. Setup Device and Model
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    
    speech_branch = SpeechEmotionModel(input_dim=40, hidden_dim=64, num_layers=2, num_classes=7)
    vocab_size = len(word_to_idx)
    text_branch = TextEmotionModel(vocab_size=vocab_size, embedding_dim=64, hidden_dim=64, num_layers=2, num_classes=7)
    
    model = MultimodalFusionModel(speech_model=speech_branch, text_model=text_branch, fusion_dim=128, num_classes=7).to(device)
    
    pipeline_dir = os.path.dirname(os.path.abspath(__file__))
    model_save_path = os.path.join(pipeline_dir, "best_fusion_model.pth")
    
    if not os.path.exists(model_save_path):
        print(f"Error: Model checkpoint {model_save_path} not found. Please train the model first.")
        return
        
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.eval()
    
    # 3. Inference Loop
    all_preds = []
    all_labels = []
    all_fused_reps = []
    all_speech_reps = []
    all_text_reps = []
    all_words = []
    all_paths = []
    
    with torch.no_grad():
        for batch in test_loader:
            speech = batch["speech"].to(device)
            text = batch["text"].to(device)
            labels = batch["label"].to(device)
            words = batch["word"]
            paths = batch["path"]
            
            # Extract intermediate representations from temporal, contextual and fusion blocks
            fused_feat, speech_feat, text_feat = model.extract_fusion_features(speech, text)
            outputs = model(speech, text)
            
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_fused_reps.extend(fused_feat.cpu().numpy())
            all_speech_reps.extend(speech_feat.cpu().numpy())
            all_text_reps.extend(text_feat.cpu().numpy())
            all_words.extend(words)
            all_paths.extend(paths)
            
    # 4. Metrics
    test_acc = accuracy_score(all_labels, all_preds)
    print(f"\nMultimodal Fusion Model Test Accuracy: {test_acc*100:.2f}%")
    
    print("\nClassification Report:")
    report_dict = classification_report(all_labels, all_preds, target_names=EMOTIONS, output_dict=True)
    report_str = classification_report(all_labels, all_preds, target_names=EMOTIONS)
    print(report_str)
    
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)
    
    # Create Results dir if not exists
    results_dir = os.path.join(project_dir, "Results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Save results to a pickle for comparison plotting
    results_save_path = os.path.join(results_dir, "fusion_results.pkl")
    with open(results_save_path, "wb") as f:
        pickle.dump({
            "test_accuracy": test_acc,
            "predictions": all_preds,
            "labels": all_labels,
            "confusion_matrix": cm,
            "report_dict": report_dict,
            "report_str": report_str,
            "learned_representations": all_fused_reps,
            "speech_representations": all_speech_reps,
            "text_representations": all_text_reps,
            "words": all_words,
            "paths": all_paths
        }, f)
        
    print(f"\n✅ Multimodal Fusion testing completed. Results saved to: {results_save_path}")

if __name__ == "__main__":
    test()
