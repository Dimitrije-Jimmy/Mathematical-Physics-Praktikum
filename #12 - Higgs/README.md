# Higgs Boson Classification with Machine Learning

This project explores machine learning techniques for distinguishing Higgs boson events from background processes using simulated data from the ATLAS experiment.

## Overview
- Implemented both **Deep Neural Networks (DNNs)** and **Boosted Decision Trees (BDTs)** for classification.
- Evaluated various hyperparameters, activation functions, and optimizers.
- Used **TensorFlow (Keras)** and **PyTorch** for DNNs, and **CatBoost** for BDTs.
- Analyzed preprocessing, feature selection, and model performance using **ROC curves**.

## Highlights
- **Preprocessing:** Normalized features, compared derived vs. raw variables.
- **TensorFlow Challenges:** GPU incompatibility on Windows, leading to a PyTorch reimplementation.
- **BDT Superiority:** Achieved competitive results with decision trees, often outperforming DNNs in classification accuracy.
- **Key Findings:** ELU activation and Adam optimizer worked best for DNNs; boosting significantly improved BDT performance.

## Dependencies
- Python 3.x
- TensorFlow, PyTorch, CatBoost
- NumPy, Pandas, Scikit-learn, Matplotlib

## License

The contents of the repository are licensed under a [MIT License][MIT].

[![MIT 4.0][MIT-shield]][MIT-sa] 

[MIT]: [http://creativecommons.org/licenses/by-nc-sa/4.0/](https://opensource.org/license/mit)
[MIT-shield]: [https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg](https://img.shields.io/badge/license-MIT-blue)
