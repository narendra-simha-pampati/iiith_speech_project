import torch
import torch.nn as nn

class SpeechEmotionModel(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=64, num_layers=2, num_classes=7):
        super(SpeechEmotionModel, self).__init__()
        # Input shape: [batch, time_steps, input_dim]
        # Temporal Modelling: Bidirectional GRU
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if num_layers > 1 else 0.0
        )
        # Average pooling over time steps is applied in extract_features.
        # Representation size: hidden_dim * 2 = 128
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
    def extract_features(self, x):
        """
        Extracts temporal representation from the GRU (Temporal Modelling block output).
        Shape: [batch, hidden_dim * 2] (size 128)
        """
        out, _ = self.gru(x) # [batch, time_steps, hidden_dim * 2]
        feat = torch.mean(out, dim=1) # [batch, hidden_dim * 2]
        return feat
        
    def forward(self, x):
        feat = self.extract_features(x)
        logits = self.fc(feat)
        return logits


class TextEmotionModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=64, num_layers=2, num_classes=7):
        super(TextEmotionModel, self).__init__()
        # Feature Extraction: Embedding layer (tokens x features)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        # Contextual Modelling: Bidirectional GRU
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if num_layers > 1 else 0.0
        )
        # Representation size: hidden_dim * 2 = 128
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
    def extract_features(self, x):
        """
        Extracts contextual representation from the GRU (Contextual Modelling block output).
        Shape: [batch, hidden_dim * 2] (size 128)
        """
        emb = self.embedding(x) # [batch, tokens, embedding_dim]
        out, _ = self.gru(emb) # [batch, tokens, hidden_dim * 2]
        feat = torch.mean(out, dim=1) # [batch, hidden_dim * 2]
        return feat
        
    def forward(self, x):
        feat = self.extract_features(x)
        logits = self.fc(feat)
        return logits


class MultimodalFusionModel(nn.Module):
    def __init__(self, speech_model, text_model, fusion_dim=128, num_classes=7):
        super(MultimodalFusionModel, self).__init__()
        self.speech_branch = speech_model
        self.text_branch = text_model
        
        # Early Fusion Concatenation: 128 (Speech) + 128 (Text) = 256
        self.fusion_layer = nn.Sequential(
            nn.Linear(128 + 128, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        # Classifier from fusion block
        self.classifier = nn.Linear(fusion_dim, num_classes)
        
    def extract_fusion_features(self, speech_x, text_x):
        """
        Extracts temporal representation (speech), contextual representation (text),
        and fusion representation (multimodal).
        """
        speech_feat = self.speech_branch.extract_features(speech_x) # [batch, 128]
        text_feat = self.text_branch.extract_features(text_x)     # [batch, 128]
        
        # Early Fusion (Concatenation)
        concat_feat = torch.cat((speech_feat, text_feat), dim=1)  # [batch, 256]
        
        # Fusion MLP
        fused_feat = self.fusion_layer(concat_feat) # [batch, 128]
        return fused_feat, speech_feat, text_feat
        
    def forward(self, speech_x, text_x):
        fused_feat, _, _ = self.extract_fusion_features(speech_x, text_x)
        logits = self.classifier(fused_feat)
        return logits
