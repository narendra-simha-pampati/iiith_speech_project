import os
import pickle
import sys

# Ensure project directory is in import path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project.utils import get_dataset_df, get_word_disjoint_splits, build_vocab, cache_dataset_features

def main():
    print("=== TESS Multimodal Feature Extraction & Caching ===")
    
    # 1. Discover dataset
    df = get_dataset_df()
    print(f"Discovered {len(df)} unique records in the dataset.")
    
    if len(df) == 0:
        print("Error: No dataset found. Please check TESS directory path.")
        return
        
    # 2. Build vocabulary
    word_to_idx, idx_to_word = build_vocab(df)
    print(f"Built text vocabulary of size: {len(word_to_idx)} words.")
    
    # 3. Create disjoint word split (Train: 70%, Val: 15%, Test: 15%)
    train_df, val_df, test_df = get_word_disjoint_splits(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42)
    print(f"Data Splits (Disjoint target words):")
    print(f"  Train: {len(train_df)} samples ({train_df['word'].nunique()} unique words)")
    print(f"  Val:   {len(val_df)} samples ({val_df['word'].nunique()} unique words)")
    print(f"  Test:  {len(test_df)} samples ({test_df['word'].nunique()} unique words)")
    
    # 4. Extract and cache features
    print("\n--- Processing Training Set ---")
    train_samples = cache_dataset_features(train_df, word_to_idx)
    
    print("\n--- Processing Validation Set ---")
    val_samples = cache_dataset_features(val_df, word_to_idx)
    
    print("\n--- Processing Test Set ---")
    test_samples = cache_dataset_features(test_df, word_to_idx)
    
    # 5. Save everything to a single pickle file
    cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cached_data.pkl")
    print(f"\nSaving preprocessed features to: {cache_path}")
    cache_payload = {
        "train_samples": train_samples,
        "val_samples": val_samples,
        "test_samples": test_samples,
        "word_to_idx": word_to_idx,
        "idx_to_word": idx_to_word
    }
    with open(cache_path, "wb") as f:
        pickle.dump(cache_payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    print("✅ Feature extraction and caching completed successfully!")

if __name__ == "__main__":
    main()
