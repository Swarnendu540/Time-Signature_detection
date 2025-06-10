🕒 Time Signature Detection (Classification of 3/4, 4/4, 5/4, 7/4)
📜 Abstract

This project presents a deep learning-based approach to automatically detect time signatures (meters) from musical audio files. The task is framed as a 4-class classification problem involving meters 3/4, 4/4, 5/4, and 7/4. Using a curated dataset of 2800 labeled audio files, we extract mel spectrogram features and train models including CNNs, ResNet-18, and EfficientNet to classify the rhythmic structure. Accurate detection of musical meter is critical in music theory analysis, music education, AI-based composition, and indexing of large-scale music databases.
📘 Introduction

In musical analysis, the time signature (or meter) is a fundamental property that defines the rhythmic organization of a piece. Despite being vital for tasks like automatic transcription or accompaniment, meter is often overlooked in machine learning pipelines, especially when dealing with raw audio. This project aims to fill this gap by developing a robust, scalable pipeline for time signature detection from real audio recordings.
📚 Background

While much of the literature in Music Information Retrieval (MIR) focuses on tasks like genre classification or mood detection, rhythmic structure classification is less explored due to its complexity. Unlike symbolic data (e.g., MIDI), raw audio presents challenges such as tempo variation, expressive timing, and acoustic noise. This project builds on techniques from MIR and deep learning to detect rhythmic structure in raw audio using supervised classification.
💡 Motivation

Traditional tools either assume known time signatures or rely on symbolic formats. In many real-world contexts (e.g., YouTube recordings, live performances), only raw audio is available. Hence, a deep learning system capable of identifying time signatures from such data can significantly aid:

    Music educators

    Composers

    Musicologists

    Digital music libraries

🎯 Objectives

    Detect 3/4, 4/4, 5/4, and 7/4 meters from raw audio.

    Evaluate multiple deep learning models, including EfficientNet.

    Provide a reusable, scalable codebase for future research and industry applications.

📁 Dataset Description

    Name: Meter 2800

    Size: 2800 audio files

    Format: WAV (converted from MP3)

    Labels: 3/4, 4/4, 5/4, 7/4

    Sources: FMA, MAG, and OWN collections

    Splits: Training, Validation, and Test

    Augmentation: Time-stretching, pitch-shifting (using audiomentations)

🧮 Features

    Mel Spectrograms: Captures time-frequency dynamics

    Sample Rate: 22,050 Hz standardized

    Window Size & Hop Length: Optimized for rhythmic cues

    Augmented Features: Improved generalization

🧹 Data Preprocessing

    Mount and extract compressed MP3 datasets

    Convert to WAV using pydub

    Apply augmentation (tempo, pitch)

    Extract mel spectrograms via librosa

    Create CSV metadata for training

🤖 Models Deployed
✅ CNN (Custom)

    2D convolution layers

    BatchNorm + ReLU

    Best for simple baseline

✅ ResNet-18

    Pretrained on ImageNet

    Transfer learning via finetuning

    Good balance between accuracy and speed

✅ EfficientNet-B0

    State-of-the-art accuracy/parameter tradeoff

    Handles deeper representations with fewer resources

    Best performance on test data

🛠️ Tools and Libraries

    Python, Pandas, NumPy

    Librosa, Pydub, Audiomentations

    PyTorch, Torchvision

    Matplotlib, Seaborn

    Google Colab with GPU (T4)

🧪 Training Process

    Optimizer: Adam

    Learning Rate: 1e-3 with scheduler

    Batch Size: 64

    Epochs: ~25

    Loss: CrossEntropy

    AMP (Automatic Mixed Precision) for faster GPU training

📊 Results Summary
Model	Accuracy	F1 Score	Comments
CNN (Custom)	~75%	Moderate	Baseline performance
ResNet-18	~82%	Good	Generalizes well
EfficientNet	~87%	Best	Most accurate and efficient
📉 Evaluation Metrics

    Accuracy

    Class-wise Precision & Recall

    Macro/Micro F1-Score

    Confusion Matrices (per epoch)

    Loss & Accuracy Curves

📈 Performance Analysis

    EfficientNet outperformed other models on all metrics.

    Class imbalance (fewer 5/4 & 7/4 examples) led to minor misclassifications.

    Augmentation helped reduce overfitting and improved robustness.

⚠️ Limitations

    Dataset is relatively small for deep networks.

    Real-world recordings may have background noise and tempo shifts.

    Ambiguous transitions between meters remain challenging.

✅ Conclusion

We demonstrate a successful pipeline for audio-based time signature detection. Among various models, EfficientNet provides the most accurate results. With proper preprocessing and feature extraction, detecting rhythmic structure from raw audio is feasible and scalable.
🔑 Key Findings

    Mel spectrograms are effective features for capturing rhythmic structure.

    EfficientNet offers the best accuracy-complexity tradeoff.

    Audio augmentation boosts generalization and class separation.

🌍 Real-life Applications

This project enables real-time or offline classification of musical meters, which has applications in:

    Music Education: Automated rhythm feedback tools

    Music Production: Smart DAWs that adapt to rhythm

    Music Retrieval: Search by time signature

    Musicological Research: Analyzing temporal evolution of musical forms

🔭 Future Work

    Integrate beat tracking or tempo curves for hybrid features

    Expand dataset to more meters (e.g., 6/8, 9/8, compound meters)

    Experiment with transformer models for sequence prediction

    Build a web app for interactive meter detection

