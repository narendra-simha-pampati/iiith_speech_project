# Multimodal Fusion Error Analysis Report

Here are 5 representative failure cases from the Multimodal Fusion model showing acoustic confusions in TESS:

| No. | Target Word | True Emotion | Predicted Emotion | Audio File Name | Failure Cause & Acoustic Analysis |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **1** | `bean` | `fear` | `happy` | `OAF_bean_fear.wav` | High pitch peak. The actor's trembling high-pitch vocal energy during extreme fear mimicked excited happy expressions. |
| **2** | `late` | `happy` | `ps` | `YAF_late_happy.wav` | Rapid articulation and high pitch. The happy tone was confused with sudden, excited pleasant surprise. |
| **3** | `pick` | `disgust` | `sad` | `OAF_pick_disgust.wav` | Slow pace and low energy. The low-arousal vocal profile of disgust matched the flat, low pitch contours typical of sadness. |
| **4** | `tool` | `fear` | `ps` | `YAF_tool_fear.wav` | Breathy, whispery speech delivery. The whispery vocal qualities of fear matched the acoustic signatures of pleasant surprise. |
| **5** | `bought` | `sad` | `neutral` | `OAF_bought_sad.wav` | Moderate tempo with flat pitch. The soft delivery of sadness lay very close to the steady, unaroused neutral baseline. |
