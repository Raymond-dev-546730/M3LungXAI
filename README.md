# M³LungXAI: Explainable Multimodal Lung Cancer Diagnosis

This repository contains the complete implementation, training scripts, evaluation code, and deployment system from:

> **"M³LungXAI: Deployable Explainable Multimodal Lung Cancer Diagnosis Through Late Fusion of CT, X-ray, and Clinical Symptoms"**  

## Overview

M³LungXAI is the first fully open-source multimodal explainable lung cancer diagnostic system that integrates CT imaging, chest X-ray, and clinical symptom extraction through a modular late fusion architecture. The system achieves high diagnostic performance while operating under real-world hardware constraints (2.0 GB VRAM), enabling deployment on standard clinical workstations in community hospitals without cloud dependencies.

**Key Features:**
- **CT Modality:** 95.88% accuracy, 99.63% ROC AUC (MobileNetV2)
- **X-ray Modality:** 91.33% accuracy, 93.18% ROC AUC (Gradient Boosting ensemble)
- **Symptom Modality:** 99.76% F1 score, 98.86% ROC AUC (SpanBERT NER)
- **Hardware Efficiency:** Runs on 2.0 GB VRAM with throughput of 36-290 samples/second
- **Explainability:** Instance-level visualisations for clinical trust and verification

## Repository Contents
~~~
Deployment/
├── DEPLOY_CT_SCAN.py                    # CT scan inference pipeline with LIME/Grad-CAM++ explainability
├── DEPLOY_MULTIMODAL.py                 # Multimodal late fusion system with clinical web interface
├── DEPLOY_SYMPTOM.py                    # Symptom extraction inference with SpanBERT NER
├── DEPLOY_X_RAY.py                      # X-ray ensemble inference pipeline with LIME explainability
├── PROCESS_CT_SCAN.py                   # CT scan preprocessing
├── PROCESS_X_RAY.py                     # X-ray preprocessing
└── SYMPTOM_DATASET.json                 

Models/
├── CT_scan_Modality/
│   ├── Scripts/
│   │   ├── Inference_Evaluation/
│   │   │   └── MobileNetV2_CT_scan_ROC.py              # ROC curve generation and hardware benchmarking
│   │   ├── Preprocessing/
│   │   │   └── CT_scan_Preprocessing.py                # CT scan preprocessing pipeline for training
│   │   └── Training/                                   # 5-fold cross-validation training for 5 CNN architectures
│   │       ├── DenseNet121_CT_scan.py
│   │       ├── MobileNetV2_CT_scan.py
│   │       ├── ResNet50_CT_scan.py
│   │       ├── SqueezeNet_1.1_CT_scan.py
│   │       └── VGG19_CT_scan.py
│   └── Terminal_Logs/                                  # Complete training and inference terminal outputs
│       ├── Inference_Evaluation/
│       │   └── MobileNetV2 CT scan ROC Terminal Logs.txt
│       └── Training/
│           ├── DenseNet121 CT scan Terminal Logs.txt
│           ├── MobileNetV2 CT scan Terminal Logs.txt
│           ├── ResNet50 CT scan Terminal Logs.txt
│           ├── SqueezeNet 1.1 CT scan Terminal Logs.txt
│           └── VGG19 CT scan Terminal Logs.txt
│
├── X-ray_Modality/
│   ├── Scripts/
│   │   ├── Inference_Evaluation/
│   │   │   └── Ensemble_Model_CXR14_ROC.py             # Final ensemble ROC generation and hardware benchmarking
│   │   ├── Preprocessing/
│   │   │   └── CXR14_Preprocessing.py                  # X-ray preprocessing pipeline for training
│   │   └── Training/
│   │       ├── Ensemble_Experiments/                   # Stacking experiments: averaging and meta-stacking
│   │       │   ├── Averaged_Ensemble_CXR14.py
│   │       │   ├── Averaged_Meta-Models-Ensemble_CXR14.py
│   │       │   └── LogisticRegression_Meta-Models-Ensemble_CXR14.py
│   │       ├── Final_Model/
│   │       │   └── Ensemble_Model_CXR14.py             # Selected Gradient Boosting meta-learner (base features)
│   │       ├── Stage1_Base_CNNs/                       # 4 diverse CNN architectures with different seeds
│   │       │   ├── ConvNeXtTiny_CXR14.py
│   │       │   ├── DenseNet121_CXR14.py
│   │       │   ├── EfficientNetV2-S_CXR14.py
│   │       │   └── ResNet18_CXR14.py
│   │       └── Stage2_Meta_Learners/
│   │           ├── Base_Features/                      # 5 meta-learners trained on 4D CNN probabilities
│   │           │   ├── GradientBoosting_Base-features_Ensemble_CXR14.py
│   │           │   ├── LogisticRegression_Base-features_Ensemble_CXR14.py
│   │           │   ├── MLP_Base-features_Ensemble_CXR14.py
│   │           │   ├── RandomForest_Base-features_Ensemble_CXR14.py
│   │           │   └── SVM_Base-features_Ensemble_CXR14.py
│   │           └── Engineered_Features/                # 5 meta-learners trained on 11D engineered features
│   │               ├── GradientBoosting_Engineered-features_Ensemble_CXR14.py
│   │               ├── LogisticRegression_Engineered-features_Ensemble_CXR14.py
│   │               ├── MLP_Engineered-features_Ensemble_CXR14.py
│   │               ├── RandomForest_Engineered-features_Ensemble_CXR14.py
│   │               └── SVM_Engineered-features_Ensemble_CXR14.py
│   └── Terminal_Logs/                                  # Complete training and inference terminal outputs
│       ├── Inference_Evaluation/
│       │   └── Ensemble Model CXR14 ROC Terminal Logs.txt
│       └── Training/
│           ├── Ensemble_Experiments/
│           │   ├── Averaged Ensemble CXR14 Terminal Logs.txt
│           │   ├── Averaged Meta-Models Ensemble CXR14 Terminal Logs.txt
│           │   └── LogisticRegression Meta-Models Ensemble CXR14 Terminal Logs.txt
│           ├── Final_Model/
│           │   └── Ensemble Model CXR14 Terminal Logs.txt
│           ├── Stage1_Base_CNNs/
│           │   ├── ConvNeXtTiny CXR14 Terminal Logs.txt
│           │   ├── DenseNet121 CXR14 Terminal Logs.txt
│           │   ├── EfficientNetV2-S CXR14 Terminal Logs.txt
│           │   └── ResNet18 CXR14 Terminal Logs.txt
│           └── Stage2_Meta_Learners/
│               ├── Base_Features/
│               │   ├── GradientBoosting Base-features Ensemble CXR14 Terminal Logs.txt
│               │   ├── LogisticRegression Base-features Ensemble CXR14 Terminal Logs.txt
│               │   ├── MLP Base-features Ensemble CXR14 Terminal Logs.txt
│               │   ├── RandomForest Base-features Ensemble CXR14 Terminal Logs.txt
│               │   └── SVM Base-features Ensemble CXR14 Terminal Logs.txt
│               └── Engineered_Features/
│                   ├── GradientBoosting Engineered-features Ensemble CXR14 Terminal Logs.txt
│                   ├── LogisticRegression Engineered-features Ensemble CXR14 Terminal Logs.txt
│                   ├── MLP Engineered-features Ensemble CXR14 Terminal Logs.txt
│                   ├── RandomForest Engineered-features Ensemble CXR14 Terminal Logs.txt
│                   └── SVM Engineered-features Ensemble CXR14 Terminal Logs.txt
│
└── Symptom_Modality/ 
    ├── Scripts/
    │   ├── Inference_Evaluation/
    │   │   └── SM_ROC.py                               # ROC curve generation for token-level NER and hardware benchmarking
    │   ├── Synthetic_Notes_Generation/                 # Complete pipeline to generate synthetic clinical notes
    │   │   ├── Generate_Clinical_Notes_SM.py           # Generates complete synthetic clinical notes dataset with LLM and validation
    │   │   ├── patient_names.json                      
    │   │   ├── symptom_constraints.json                
    │   │   └── symptom_synonyms.json                   # 13-20 variants per symptom for linguistic diversity
    │   └── Training/
    │       └── Train_SM.py                             # SpanBERT Large fine-tuning for token-level NER
    └── Terminal_Logs/                                  # Complete training and inference terminal outputs
        ├── Inference_Evaluation/
        │   └── SM ROC Terminal Logs.txt
        └── Training/
            └── Train SM Terminal Logs.txt
~~~
