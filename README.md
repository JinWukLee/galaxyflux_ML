# Predicting Galaxy Flux with Machine Learning
This project explores the use of convolutional neural networks (CNNs) to predict galaxy flux from 2D grayscale images, using simulated data inspired by the Hyper Suprime-Cam Survey. We benchmarked traditional regression methods and developed a deep learning model that significantly outperformed linear and tree-based approaches in accuracy and scalability.

> Conducted as a final project for PHYS 378 at Yale University  
> Collaborators: Jin Wuk Lee & Jinwoo Kim  
> Focus Areas: Machine Learning, Model Evaluation

## Background
**Galaxy flux** measures the total light received from a galaxy—a fundamental quantity for astronomers. Traditional methods like light-profile fitting are computationally intensive and do not scale to modern survey sizes.

This project:
- Uses 40,000 simulated galaxy images with associated metadata
- Tests linear regression, logistic models, and random forest as baselines
- Implements 3 CNN model to predict log-scaled flux values from image tensors

## Model Summary
- **Baseline Models**:  
  - Linear Regression → MSE ≈ 0.178  
  - Logistic/Random Forest → Accuracy ≈ 34%

- **CNN Architecture**:  
  - 2 Convolutional layers → ReLU + MaxPooling  
  - Dense → Dropout(0.2) → Regression output  
  - Optimizer: Adam  
  - Best Config: `lr=0.001`, `epochs=10`  
  - **Best CNN Performance**:  
    - MSE (Train): `0.0815`  
    - MSE (Test): `0.0489`

## Dataset
- **Images**: 40,000 grayscale `.png` files, each 143×143 pixels  
- **Metadata Columns** (17 total): 
  - e.g. `flux`, `Bt` (bulge-to-total), `Re`, `PA`, `axis_ratio`, etc.
- **Files Included**:
  - `train.csv`, `test.csv`: metadata for 30k/10k samples
  - `tensors.csv`, `tensors_df.pkl`: preprocessed image data for faster loading

## Repository Structure
galaxy-flux-ml/
├── Predicting_Galaxy_Flux_with_ML_.ipynb # Jupyter notebook (end-to-end code)
├── PHYS378_Final_Project__Predicting_Galaxy_Flux_with_ML.pdf # Final report
├── train.csv # Training metadata
├── test.csv # Testing metadata
├── tensors.csv # Image tensors (flattened)
├── tensors_df.pkl # Pandas dataframe of tensors
├── Final Report PDF

## Results Summary
| Model      | MSE (Train) | MSE (Test) | Notes                          |
|------------|-------------|------------|--------------------------------|
| Linear     | 0.17889     | 0.17867    | Baseline, poor correlation     |
| CNN (Best) | 0.0815      | 0.0489     | `lr=0.001`, `epochs=10`        |
| CNN (Overfit) | 0.0561   | 0.0388     | Slight signs of overfitting    |

## Future Work
- Integrate metadata like `Bt`, `Re` into hybrid CNN + dense input
- Perform dimensionality reduction (PCA)
- Use real survey data from SDSS or Hyper Suprime-Cam
- Deploy as a web-based regression demo (Streamlit or Gradio)




