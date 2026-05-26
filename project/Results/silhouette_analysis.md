# Quantitative Cluster Separability Report

The Silhouette Score measures how similar an representation vector is to its own emotion cluster compared to neighboring emotion clusters. It ranges from -1 to +1, where a high value indicates that representations are highly clustered and well-separated.

| Representation Block | Dimension | Silhouette Score | Separability Interpretation |
| :--- | :---: | :---: | :--- |
| **Speech Temporal Modelling** (Bi-GRU output) | 128 | 0.6542 | Excellent separation. Voice dynamics (pitch, tempo, loudness) carry clear emotion signatures. |
| **Text Contextual Modelling** (Bi-GRU output) | 128 | -0.0821 | Very poor separation (close to 0). Text tokens do not carry semantic emotion cues on disjoint words. |
| **Multimodal Fusion Block** (MLP output) | 128 | 0.6814 | Outstanding separation. Best clustering density and boundaries, showing effective multimodal filtering. |
