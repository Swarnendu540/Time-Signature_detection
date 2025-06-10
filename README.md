📌 Abstract

This project explores the automatic detection of musical time signatures from raw audio recordings. Framed as a 4-class classification problem (3/4, 4/4, 5/4, 7/4), it leverages deep learning techniques to extract rhythmic patterns using mel spectrograms and classify them using models such as Custom CNNs, ResNet-18, and EfficientNet-B0. We work with a curated dataset of 2800 audio samples, applying robust preprocessing, augmentation, and training strategies to achieve high classification accuracy. This pipeline contributes significantly to music information retrieval, AI-assisted music tools, and educational software.
📖 Table of Contents

    Introduction

    Background

    Motivation

    Objectives

    Dataset Description

    Feature Engineering

    Data Preprocessing

    Models

    Tools and Libraries

    Training Process

    Results

    Evaluation Metrics

    Performance Analysis

    Limitations

    Conclusion

    Key Findings

    Real-World Applications

    Future Work

    Project Structure

📘 Introduction

Time signature, or meter, is the rhythmic foundation of music. Automatically identifying it from raw audio enables intelligent music processing, understanding, and composition. This project builds a pipeline to detect time signatures directly from WAV files, using modern deep learning methods.
🧠 Background

Traditional music analysis focuses on symbolic representations like MIDI. However, real-world applications often involve raw, noisy audio. In this project, we use deep learning on mel spectrograms to approximate temporal rhythmic structure and identify meter without relying on symbolic data.
💡 Motivation

Most music AI systems skip over time signature analysis or depend on symbolic input. We aim to:

    Detect rhythmic structure directly from raw audio

    Assist educators, composers, and researchers

    Improve MIR tools with accurate time signature tagging

🎯 Objectives

    Build a robust classifier for time signatures (3/4, 4/4, 5/4, 7/4)

    Apply data augmentation to increase robustness

    Evaluate different deep learning models

    Visualize and analyze classification results

📁 Dataset Description

    Total Samples: 2800 WAV files

    Time Signatures: 3/4, 4/4, 5/4, 7/4

    Sources: FMA, MAG, OWN (converted from MP3)

    Preprocessing: MP3 ➝ WAV ➝ Mel Spectrograms

    Augmentation: Time-stretching, pitch-shifting via audiomentations

🎼 Feature Engineering

    Mel Spectrograms extracted using librosa

    128 mel bands with time-frequency structure

    Fixed-length windows for consistent input

🧹 Data Preprocessing

    Unzip and convert MP3s to WAV

    Apply augmentation (pitch, tempo)

    Extract mel spectrograms

    Save data to CSV with labels

    Visualize sample spectrograms

🧠 Models
✅ Custom CNN

    Simple 2D CNN with BatchNorm + ReLU

    Lightweight and easy to train

✅ ResNet-18

    Pretrained on ImageNet

    Finetuned for 4-class classification

✅ EfficientNet-B0 ⭐

    Most accurate model

    High efficiency with low parameter count

    Best performance on test set

🛠️ Tools and Libraries

    Python, NumPy, Pandas

    Librosa, Pydub, Audiomentations

    PyTorch, Torchvision

    Matplotlib, Seaborn

    Google Colab (T4 GPU + AMP)

🧪 Training Process

    Optimizer: Adam

    Loss: CrossEntropyLoss

    Epochs: ~25

    Batch Size: 64

    Mixed Precision (AMP): Enabled

    Train/Validation/Test Split: Stratified

📊 Results
Model	Accuracy	F1 Score	Notes
CNN (Custom)	~75%	Moderate	Good for baseline
ResNet-18	~82%	Strong	Transfer learning works well
EfficientNet	~87%	Best	Most accurate and efficient
📉 Evaluation Metrics

    Accuracy

    Precision / Recall / F1-Score

    Class-wise Performance

    Confusion Matrix (per epoch)

    Loss & Accuracy Curves

📈 Performance Analysis

    EfficientNet outperformed all models.

    3/4 and 4/4 sometimes confused due to similar rhythm.

    Augmentation greatly improved generalization.

    Minor class imbalance affected 5/4 and 7/4 accuracy.

⚠️ Limitations

    Dataset size is moderate for deep learning.

    Ambiguity in rhythm for closely spaced classes.

    Augmentation cannot fully mimic real-world complexity.

✅ Conclusion

We successfully built a deep learning pipeline to classify time signatures from raw audio. EfficientNet-B0 emerged as the best model. This work shows that rhythm classification from raw signals is not only feasible but can be applied to real-world music systems.
🔑 Key Findings

    Mel spectrograms are suitable features for meter detection.

    EfficientNet achieves high accuracy with fewer resources.

    Augmentation techniques improve model generalization.

🌍 Real-World Applications

    🎓 Music Education: Auto-evaluation tools for rhythm training

    🧠 MIR Systems: Improved search and tagging by meter

    🎼 Composition Tools: AI-assisted music generators with rhythm awareness

    🔎 Musicology: Quantitative rhythm analysis of large corpora

🔭 Future Work

    Add compound and irregular time signatures (e.g., 6/8, 9/8, 11/8)

    Explore beat tracking + spectrogram fusion

    Experiment with transformer-based audio models

    Deploy as a web app or VST plugin

