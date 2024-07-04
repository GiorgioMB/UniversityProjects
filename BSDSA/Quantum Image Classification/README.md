# Quantum Image Classification Project - BSDSA
## Overview
This repository houses a project that explores the application of quantum computing techniques for the classification of binary images with vertical and horizontal stripes. Utilizing quantum circuits, this project aims to process and classify binary image data, demonstrating the potential benefits of quantum superposition and entanglement in achieving effective classification with a compact feature map.

## Project Structure
This project includes comprehensive Python scripts that address all critical phases from data generation to model evaluation:

### ImageClassification.py
- **Purpose**: Implements a quantum classifier to differentiate between horizontal and vertical stripes in binary images.
- **Key Features**:
  - Generates synthetic binary image data with designated stripe patterns.
  - Constructs and utilizes quantum circuits for image data encoding and classification.
  - Trains quantum models to classify images and evaluates their performance.
  - Visualizes both the generated data and the performance metrics of quantum models.
  - Includes a classical neural network for performance comparison.
### Performance
- **Effectiveness**: Following the configurations outlined in `ImageClassification.py`, the quantum circuit consistently demonstrates performance on par with, or even exceeding, that of a conventional linear layer. This highlights the quantum model's potential as a robust alternative for processing complex image data.

## Libraries Used
- PennyLane
- NumPy
- Matplotlib
- Seaborn
- PyTorch

## How to Run
Before running the script, make sure you have installed Python and the required libraries. Here are the steps:

1. Clone or download `ImageClassification.py` to your local machine.
2. Navigate to the directory where you saved the script via terminal or command prompt.
3. Adjust script parameters (e.g., number of qubits, image size, epochs) as needed.
4. Run the script with the following command:
   ```bash
   python ImageClassification.py
   ```
   
## Future Expansions
In collaboration with the Bocconi Students Data Science Association (BSDSA) we aim to further enhance the capabilities of the quantum image classifier through:
- Expanding to New Datasets: Applying the developed techniques to different datasets to assess and refine the robustness and utility of our quantum classification approach.
- Scholarly Publication: Documenting our methodologies and findings in a detailed paper
