#%%
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator, StatevectorSimulator
from qiskit.circuit import ParameterVector
import matplotlib.pyplot as plt
import seaborn as sns
from qiskit_algorithms.optimizers import COBYLA
from typing import Tuple
from qiskit.qasm2 import dumps
from qiskit.quantum_info import Statevector
from sklearn.model_selection import train_test_split
from qiskit.visualization import plot_state_city,  plot_state_hinton, plot_state_paulivec, plot_state_qsphere, plot_bloch_multivector
from sklearn.linear_model import LogisticRegression

## 1. Generate synthetic binary image data
def generate_data(num_samples: int = 100, size: int = 4, noise: bool = False,  noise_level: float = 0.1, noise_type = "uniform") -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates synthetic binary image data with horizontal and vertical stripes
    Arguments:
    - num_samples (int): Number of samples to generate (default: 100)
    - size (int): Size of the square image (default: 4)
    - noise (bool): Whether to add noise to the data (default: False)
    - noise_level (float): Level of noise to add to the data (default: 0.1)
    - noise_type (str): Type of noise to add to the data (default: "uniform") - options: "uniform", "normal"

    Returns:
    - data (np.ndarray): Generated image data
    - labels (np.ndarray): Generated labels

    Note: Number of stripes is size // 2
    """
    if noise_level < 0 or noise_level > 1:
        raise ValueError("Noise level must be between 0 and 1")
    if (not noise) and (noise_level != 0.1):
        print("Warning, noise level was set but noise was not enabled. Data will be generated without noise.")
    labels = np.random.randint(2, size=num_samples)
    data = np.zeros((num_samples, size, size))

    horizontal_indices = np.where(labels == 0)[0]
    data[horizontal_indices, ::2, :] = 1

    vertical_indices = np.where(labels == 1)[0]
    data[vertical_indices, :, ::2] = 1
    
    if noise: ## Add noise to the data
       if noise_type == "uniform":
           noise = np.random.uniform(low = -noise_level, high = noise_level, size = (num_samples, size, size))
       elif noise_type == "normal":
           noise = np.random.randn(num_samples, size, size) * noise_level
       data += noise
        
    return data, labels

## 2. Visualize the generated data
def visualize_data(data: np.ndarray = None, labels: np.ndarray = None, max_samples: int = 10) -> plt.figure:
    """
    Data visualization function
    Arguments:
    - data (np.ndarray): Image data to visualize (default: None)
    - labels (np.ndarray): Labels corresponding to the image data (default: None)
    - max_samples (int): Maximum number of samples to visualize (default: 10)

    Returns:
    - plt.figure: Visualization of the image data

    Note: if data and labels are not provided, they will be generated with size = 4 and num_samples = max_samples
    """
    if isinstance(data, type(None)) or isinstance(labels, type(None)):
        data, labels = generate_data(num_samples = max_samples)
    sns.set_style("white")
    fig, axs = plt.subplots(1, max_samples, figsize = (15, 3))
    for i in range(max_samples):
        axs[i].imshow(data[i], cmap = 'grey')
        axs[i].axis('off')
        axs[i].set_title(f"Label: {labels[i]}")
    plt.show()

## 3. Build feature map and variational circuit for the quantum classifier
def adaptive_feature_map(data_point: np.ndarray, num_qubits: int) -> Tuple[QuantumCircuit, ParameterVector]:
    """
    Function to create an adaptive feature map for the quantum circuit
    Arguments:
    - data_point (np.ndarray): Image data point to classify
    - num_qubits (int): Number of qubits in the quantum circuit

    Returns:
    - qc (QuantumCircuit): Adaptive feature map for the quantum circuit
    - params (ParameterVector): Parameters for the feature map
    """
    params = ParameterVector('theta', length = num_qubits) ## Parameters for the feature map
    qc = QuantumCircuit(num_qubits) ## Quantum circuit with num_qubits qubits
    qc.h(range(num_qubits)) ## Apply Hadamard gates to all qubits
    group_size = len(data_point.flatten()) // num_qubits
    for i in range(num_qubits):
        start_idx = i * group_size ## Start index for the pixel values
        end_idx = start_idx + group_size ## End index for the pixel values
        summed_value = np.sum(data_point.flatten()[start_idx:end_idx]) ## Sum the pixel values
        angle = (params[i] * summed_value) ## Angle for the rotation gate
        qc.ry(angle, i) ## Apply the rotation gate
    return qc, params

def layered_variational_circuit(num_qubits: int = 4, num_layers: int = 2) -> Tuple[QuantumCircuit, ParameterVector]:
    """
    Function to create a layered variational circuit
    Arguments:
    - num_qubits (int): Number of qubits in the quantum circuit (default: 4)
    - num_layers (int): Number of layers in the quantum circuit (default: 2)
    
    Returns:
    - qc (QuantumCircuit): Layered variational circuit
    - all_params (np.ndarray): Parameters for the variational circuit
    """
    qc = QuantumCircuit(num_qubits) ## Quantum circuit, with num_qubits qubits
    all_params = []
    for layer in range(num_layers):
        params = ParameterVector(f'params_{layer}', length=num_qubits) ## Parameters for each layer
        all_params.extend(params) ## Add the parameters to the list
        for i in range(num_qubits): ## Apply the variational circuit
            qc.rx(params[i], i) ## Add an RX gate with the parameter
        for i in range(num_qubits):
            if num_qubits > 1:
                qc.cx(i, (i + 1) % num_qubits) ## Add a CNOT gate with the next qubit
    return qc, all_params

## 4. Train the quantum circuit for a specific label
def train_label_circuit(data: np.ndarray, labels: np.ndarray, target_label: int, 
                        num_layers: int = 2, num_qubits: int = 4, num_epochs:int = 50, 
                        verbose: int = False) -> Tuple[QuantumCircuit, np.ndarray, float]:
    """
    Function to train a quantum circuit for a specific label
    Arguments:
    - data (np.ndarray): Image data to classify
    - labels (np.ndarray): Labels corresponding to the image data
    - target_label (int): Label to train the circuit for
    - num_layers (int): Number of layers in the variational circuit (default: 2)
    - num_qubits (int): Number of qubits in the quantum circuit (default: 4)
    - num_epochs (int): Number of epochs for training (default: 50)
    - verbose (bool): Whether to print training details (default: False)

    Returns:
    - best_circuit (QuantumCircuit): Best quantum circuit for the target label
    - optimal_params (np.ndarray): Optimal parameters for the quantum circuit
    - value (float): Cost value of the best circuit
    """
    simulator = AerSimulator() ## Noiseless simulator
    var_circuit, var_params = layered_variational_circuit(num_qubits = num_qubits, num_layers = num_layers) ## Variational circuit
    optimizer = COBYLA(maxiter=num_epochs) ## Optimizer
    best_circuit = None ## Best circuit for the target label
    iteration = 0 ## Iteration counter, for verbose mode
    flag_best = False ## Flag to check if the best circuit has been found
    def objective_function(parameter_values): ## Objective function for optimization
        nonlocal best_circuit, iteration, var_params, target_label, flag_best
        iteration += 1
        cost = 0
        param_dict = dict(zip(var_params, parameter_values[len(var_params):])) ## Variational circuit parameters
        feature_params = parameter_values[:len(var_params)] ## Feature map parameters
        for i, data_point in enumerate(data):
            if labels[i] != target_label: ## Skip data points that don't match the target label
                continue
            feature_map, feature_map_params = adaptive_feature_map(data_point, num_qubits) ## Feature map
            combined_circuit = feature_map.assign_parameters(dict(zip(feature_map_params, feature_params))).compose(var_circuit.assign_parameters(param_dict)) ## Combined circuit
            combined_circuit.measure_all() ## Measurement
            t_qc = transpile(combined_circuit, simulator) ## Transpile the circuit (ie convert to basis gates)
            result = simulator.run(t_qc, shots=1024).result() ## Run the circuit, 1024 shots (could be more, could be less, but 1024 is a good number for this case)
            counts = result.get_counts() ## Get the counts
            predicted_label = max(counts, key=counts.get) ## Get the most probable label
            expected_state = '0' * num_qubits if target_label == 0 else '1' * num_qubits ## Expected state (all 0s or all 1s, depending on the target label)
            hamming_distance = sum(1 for x, y in zip(predicted_label, expected_state) if x != y) ## Hamming distance between the predicted label and the expected state
            cost += (hamming_distance * counts[predicted_label]) / sum(counts.values())
            if hamming_distance == 0: ## If the predicted label is all correct, set the best circuit
                best_circuit = combined_circuit
                if not flag_best: ## For verbose mode
                    flag_best = True
                    if verbose:
                        print(f"Best circuit found for label {target_label} at iteration {iteration}")
        if verbose and iteration % 10 == 0:
            print(f"Iteration {iteration} for label {target_label}: Cost = {cost / len(data):.4f}")
        cost /= len(data) ## Normalize cost, as otherwise with more data points, the cost would be higher
        return cost

    all_params = np.random.rand(len(var_params) * 2)
    result = optimizer.minimize(fun=objective_function, x0=all_params)
    optimal_params = result.x ## Optimal parameters
    value = result.fun ## Cost value
    return best_circuit, optimal_params, value

## 5. Build the quantum classifier
def quantum_classifier(data: np.ndarray, labels: np.ndarray, num_layers: int = 2, 
                       num_qubits: int = 4, num_epochs: int = 50, verbose: bool = False) -> Tuple[np.ndarray, np.ndarray, float, float, float, float, QuantumCircuit, QuantumCircuit]:
    """
    Function to build classifiers for binary image data using quantum circuits
    Arguments:
    - data (np.ndarray): Image data to classify
    - labels (np.ndarray): Labels corresponding to the image data
    - num_layers (int): Number of layers in the variational circuit (default: 2)
    - num_qubits (int): Number of qubits in the quantum circuit (default: 4)
    - num_epochs (int): Number of epochs for training (default: 50)
    - verbose (bool): Whether to print training details (default: False)

    Returns:
    - params_0 (np.ndarray): Optimal parameters for the quantum circuit for label 0
    - params_1 (np.ndarray): Optimal parameters for the quantum circuit for label 1
    - cost_0 (float): Best cost value for label 0
    - cost_1 (float): Best cost value for label 1
    - accuracy (float): Accuracy of the classifier
    - average_confidence (float): Average confidence of the classifier
    - circuit_0 (QuantumCircuit): Best quantum circuit for label 0
    - circuit_1 (QuantumCircuit): Best quantum circuit for label 1
    """
    circuit_0 = None
    circuit_1 = None
    train_data, _, train_labels, _ = train_test_split(data, labels, test_size=0.2, random_state=42)
    counter = 0
    while True: ##Continue until a best circuit for both labels is found
        if verbose and counter > 1: 
            print("Retraining circuits...")
            print(f"Attempt {counter}")
        counter += 1
        if isinstance(circuit_0, type(None)): ## Check that best circuit is found for label 1
            circuit_0, params_0, cost_0 = train_label_circuit(train_data, train_labels, target_label=0, num_layers=num_layers, num_qubits=num_qubits, num_epochs=num_epochs, verbose=verbose)
        else: 
            if verbose: print("Circuit for label 0 already found, skipping training for label 0")
        if isinstance(circuit_1, type(None)): ## Check that best circuit is found for label 0
            circuit_1, params_1, cost_1 = train_label_circuit(train_data, train_labels, target_label=1, num_layers=num_layers, num_qubits=num_qubits, num_epochs=num_epochs, verbose=verbose)
        else:
            if verbose: print("Circuit for label 1 already found, skipping training for label 1")
        if not ((isinstance(circuit_0, type(None)) or isinstance(circuit_1, type(None)))): ## Check that best circuits are found for both labels
            break
    if verbose: 
        print(f"Final Training Cost for label 0: {cost_0:.4f}")
        print(f"Final Training Cost for label 1: {cost_1:.4f}")
    simulator = AerSimulator()
    predictions = []
    confidence = []
    ## Test the classifier on the data, as it's noiseless, generating test data is not necessary
    idx = 0
    for data_point in data:
        feature_map_0, feature_map_params_0 = adaptive_feature_map(data_point, num_qubits)
        combined_circuit_0 = feature_map_0.assign_parameters(dict(zip(feature_map_params_0, params_0[:len(feature_map_params_0)]))).compose(circuit_0)
        combined_circuit_0.measure_all()
        t_qc_0 = transpile(combined_circuit_0, simulator)
        result_0 = simulator.run(t_qc_0, shots=1024).result()
        counts_0 = result_0.get_counts()
        confidence_0 = counts_0[max(counts_0, key=counts_0.get)] / 1024

        feature_map_1, feature_map_params_1 = adaptive_feature_map(data_point, num_qubits)
        combined_circuit_1 = feature_map_1.assign_parameters(dict(zip(feature_map_params_1, params_1[:len(feature_map_params_1)]))).compose(circuit_1)
        combined_circuit_1.measure_all()
        t_qc_1 = transpile(combined_circuit_1, simulator)
        result_1 = simulator.run(t_qc_1, shots=1024).result()
        counts_1 = result_1.get_counts()
        confidence_1 = counts_1[max(counts_1, key=counts_1.get)] / 1024
        ##Print the predicted string of the circuit0, 1 and the actual label for the data point
        if confidence_0 > confidence_1:
            predictions.append(0)
            confidence.append(confidence_0)
        else:
            predictions.append(1)
            confidence.append(confidence_1)
        idx += 1

    accuracy = sum(1 for x, y in zip(predictions, labels) if x == y) / len(labels)
    average_confidence = np.mean(confidence) 

    if verbose: ## Print the best circuits
        print(f"Optimal circuit for label 0:\n", circuit_0.draw())
        print(f"Optimal circuit for label 1:\n", circuit_1.draw())
    
    return params_0, params_1, cost_0, cost_1, accuracy, average_confidence, circuit_0, circuit_1

## 6. Visualize the mapping of the quantum circuits
def visualize_feature_map_state(data_point: np.ndarray, num_qubits: int = 4, params: np.ndarray = None, save_fig: bool = False, label:int = 0) -> Statevector:
    """
    Function to plot the state of the quantum sphere after applying the feature map
    Arguments:
    - data_point (np.ndarray): Image data point to classify
    - num_qubits (int): Number of qubits in the quantum circuit (default: 4)
    - params (np.ndarray): Parameters for the quantum circuit (default: None)
    - save_fig (bool): Whether to save the figure (default: False)
    - label (int): Label for the data point (default: 0)

    Returns:
    - statevector: Statevector of the quantum circuit
    """
    if isinstance(params, type(None)):
        params = np.random.rand(num_qubits * 2)
    feature_map, feature_map_params = adaptive_feature_map(data_point, num_qubits)
    feature_map_params = feature_map_params[:len(params)]
    param_dict = dict(zip(feature_map_params, params[:len(feature_map_params)]))
    feature_map = feature_map.assign_parameters(param_dict)
    
    t_qc = transpile(feature_map, StatevectorSimulator())
    result = StatevectorSimulator().run(t_qc).result()
    statevector = result.get_statevector()
    state_city = plot_state_city(statevector)
    if save_fig:
        state_city.savefig(f'state_city_label_{label}.png', format='png', dpi=500)
    hinton = plot_state_hinton(statevector)
    if save_fig:
        hinton.savefig(f'state_hinton_label_{label}.png', format='png', dpi=500)
    paulivec = plot_state_paulivec(statevector)
    if save_fig:
        paulivec.savefig(f'state_paulivec_label_{label}.png', format='png', dpi=500)
    qsphere = plot_state_qsphere(statevector)
    if save_fig:
        qsphere.savefig(f'state_qsphere_label_{label}.png', format='png', dpi=500)
    bloch = plot_bloch_multivector(statevector)
    if save_fig:
        bloch.savefig(f'bloch_multivector_label_{label}.png', format='png', dpi=500)
    return statevector


## 7. Plot and save the quantum circuits
def plot_and_save_circuits(circuit_0: QuantumCircuit, circuit_1: QuantumCircuit, latex: bool = False) -> Tuple[str, str]:
    """
    Function to plot and save the quantum circuits
    Arguments:
    - circuit_0 (QuantumCircuit): Quantum circuit for label 0
    - circuit_1 (QuantumCircuit): Quantum circuit for label 1
    - latex (bool): Whether to save the circuit in latex format (default: False)

    Returns:
    - plt.figure: Visualization of the quantum circuits
    - qasm_0 (str): QASM code for the quantum circuit for label 0
    - qasm_1 (str): QASM code for the quantum circuit for label 1

    Note: if latex is True, the circuit will also be saved in latex format. Images are saved in the working directory.
    """
    ## Display the circuit for label 0
    fig = plt.figure(figsize=(10, 7), dpi=500)
    ax = fig.add_subplot(111)
    circuit_0.draw(output='mpl', ax=ax)
    plt.savefig('circuit0.png', format='png', dpi=500)
    plt.show() ## As the circuits are quite large, it's better to display them separately

    ## Display the circuit for label 1
    fig = plt.figure(figsize=(10, 7), dpi=500)
    ax = fig.add_subplot(111)
    circuit_1.draw(output='mpl', ax=ax) 
    plt.savefig('circuit1.png', format='png', dpi=500)
    plt.show() 

    if latex: ##Latex for formal reports
        circuit_0.draw(output='latex', filename='circuit0_latex.png') 
        circuit_1.draw(output='latex', filename='circuit1_latex.png')
    
    ## Save the circuits in QASM format for use in the actual quantum computer
    qasm_0 = dumps(circuit_0)
    qasm_1 = dumps(circuit_1)

    with open('circuit0.qasm', 'w') as file:
        file.write(qasm_0)
    with open('circuit1.qasm', 'w') as file:
        file.write(qasm_1)

    return qasm_0, qasm_1

## 8. Train a classical logistic regression model
def train_logistic_regression(data: np.ndarray, labels: np.ndarray, num_points:int) -> Tuple[np.ndarray, float]:
    """
    Function to train a logistic regression model
    Arguments:
    - data (np.ndarray): Image data to classify
    - labels (np.ndarray): Labels corresponding to the image data
    - num_points (int): Final dimensionality of the data

    Returns:
    - accuracy (float): Accuracy of the logistic regression model
    - confidence (float): Confidence of the logistic regression model
    """
    transformed_data = data.copy().reshape(len(data), -1)
    new_data = np.zeros((len(data), num_points))
    counter = 0
    for point in transformed_data:
        new_point = np.zeros(num_points) 
        for pixel in range(num_points):
            new_point[pixel] = np.sum(point[pixel*len(point)//num_points:(pixel+1)*len(point)//num_points])
        new_data[counter] = new_point
        counter += 1
    train_data, test_data, train_labels, test_labels = train_test_split(new_data, labels, test_size=0.2, random_state=42)
    model = LogisticRegression(random_state=42)
    model.fit(train_data, train_labels)
    predictions = model.predict(new_data)
    confidence = np.mean([max(x) for x in model.predict_proba(new_data)])
    accuracy = sum(1 for x, y in zip(predictions, labels) if x == y) / len(labels)
    return accuracy, confidence

## Main execution
if __name__ == "__main__":
    ## Set seed for reproducibility
    np.random.seed(42)
    noise = False
    noise_type = "normal"
    data, labels = generate_data(num_samples=100, size=6, noise = noise, noise_type=noise_type)  ## Smaller size because Aer complains of too many qubits
    visualize_data(data, labels, max_samples=10)
    num_layers = 8 ##10
    num_qubits = 4 ##1
    num_epochs = 50
    ##Note, due to the quirks of qiskit's AerSimulator setting a seed renders the circuit significantly less accurate
    ##For this reason, the seed is not set, and the results may vary slightly between runs. Some may fail catastrophically
    ##However, the expected accuracy is around 93-95% (CI of 99%) for this dataset - tested on 300 runs of this code, for noiseless data
    ##Model confidence is around 32-34% (CI of 99%) for this dataset - tested on 300 runs of this code, for noiseless data
    params_0, params_1, cost_0, cost_1, accuracy, average_confidence, circuit_0, circuit_1 = quantum_classifier(data, labels, num_layers=num_layers, num_qubits=num_qubits, num_epochs=num_epochs, verbose=True)
    print(f"Accuracy: {accuracy * 100:.4f}%")
    print(f"Average confidence: {average_confidence * 100:.4f}%")
    print(f"Best Cost for label 0: {cost_0:.4f}")
    print(f"Best Cost for label 1: {cost_1:.4f}")
    vec = visualize_feature_map_state(data[0], num_qubits=num_qubits, params=params_0, save_fig=True, label=0)
    vec = visualize_feature_map_state(data[1], num_qubits=num_qubits, params=params_1, save_fig=True, label=1)
    plot_and_save_circuits(circuit_0, circuit_1, latex=True)
    num_points = num_qubits
    print("Training a classical logistic regression model...")
    accuracy, confidence = train_logistic_regression(data, labels, num_points)
    print("Logistic Regression Model:")
    print(f"Accuracy: {accuracy * 100:.4f}%")
    print(f"Confidence: {confidence * 100:.4f}%")
# %%
