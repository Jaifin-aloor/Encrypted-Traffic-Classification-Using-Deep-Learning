# Encrypted Traffic Classification Using Deep Learning

A deep learning framework to classify encrypted mobile service traffic (VPN vs. Non-VPN) using statistical flow-level features without payload inspection.

## 👥 Team Members
* **Aravindh P** 
* **Jaifin B Aloor** 
* **Richu James** 

**Institution:** Amrita School of Computing, Amrita Vishwa Vidyapeetham 
**Course:** 20CYS303 - Cybersecurity Project 

## 📌 Project Overview
This project addresses the challenge of identifying encrypted traffic patterns by implementing a complete deep learning pipeline. The system utilizes flow-based statistical features to distinguish between different traffic categories, ensuring user privacy by avoiding payload inspection.

### Phase 2 presentation: [Here](https://github.com/Jaifin-aloor/Encrypted-Traffic-Classification-Using-Deep-Learning/blob/main/Phase2.pdf)
### Phase 3 presentation: [Here](https://github.com/Jaifin-aloor/Encrypted-Traffic-Classification-Using-Deep-Learning/blob/main/Encrypted%20Traffic%20Classification%20Using%20Deep%20Learning.pdf)

### Key Features
* **Three Architectures:** Implemented CNN, LSTM, and Hybrid CNN-LSTM models.
* **Hardware Acceleration:** Optimized for Apple Silicon using the MPS (Metal Performance Shaders) backend.
* **Real-time Capabilities:** Includes a `live_demo.py` script for real-time traffic classification.
* **High Accuracy:** Achieved 77.02% accuracy across all implemented models.

## 📊 Dataset
* **Source:** ISCX VPN-NonVPN 2016 Dataset.
* **Volume:** 192,279 network flows.
* **Split:** 80% Training / 20% Testing.
* **Classes:** 14 traffic categories (7 VPN, 7 Non-VPN) including Browsing, Streaming, VoIP, and P2P.

## 🛠️ Technology Stack
* **Language:** Python 
* **Deep Learning:** PyTorch 2.x 
* **Data Processing:** NumPy, Pandas, SciPy 
* **Machine Learning:** scikit-learn 
* **Visualization:** Matplotlib 

## 📂 Repository Structure
* `data_preprocessor.py`: Handles data loading, cleaning, feature selection (8 numeric features), and standardization.
* `models.py`: Contains definitions for CNN, LSTM, and Hybrid architectures.
* `train.py`: The main training pipeline with hyperparameter tuning.
* `live_demo.py`: Script for demonstrating real-time traffic classification.
* `demo_predict.py`: Sample prediction demonstration.

## 📈 Performance Results

| Model | Accuracy | Precision | Recall | F1-Score | Training Time |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **CNN** | 77.02% | 0.65 | 0.58 | 0.61 | 4m 32s |
| **LSTM** | 77.02% | 0.65 | 0.58 | 0.61 | 5m 18s |
| **Hybrid** | 77.02% | 0.65 | 0.58 | 0.61 | 6m 45s |


* **Final Loss:** 0.4650 
* **Convergence:** Stable after epoch 15 

## 🚀 How to Run
1.  **Preprocessing:** Run `data_preprocessor.py` to load ARFF files and extract features.
2.  **Training:** Execute `train.py` to train the models (default: 20 epochs, batch size 64).
3.  **Live Demo:** Use `live_demo.py` to capture and classify real-time traffic.
