# Formal Analysis of Neural Networks for Intrusion Detection (IDS)

This repository contains the source code, experimental data, and documentation for my Undergraduate Thesis. The project focuses on applying **Formal Methods** to verify security properties in Neural Networks (MLPs) applied to cyber intrusion detection.

---

## 🚀 Getting Started

Follow the instructions below to set up the environment and reproduce the experiments.

### 1. Prerequisites
Before running the project, you need to have the following tools installed and compiled:
* **[reluka](https://github.com/spreto/reluka):** Tool used to convert ONNX models into Lukasiewicz logic formulas (`.limodsat`).
* **[lukasol](https://github.com/spreto/lukasol):** A solver for infinite-valued Lukasiewicz logic formulas.

### 2. Dependency Installation
The project uses several libraries for training the MLP neural networks. To install all requirements, run the following command in your terminal:

```bash
pip install -r requirements.txt
```

### 3. Data Download and Execution
This project uses the **CIC-IDS2017** dataset. Follow these steps to prepare the environment:

1. **Download the Dataset:** Access the cleaned version at: [CIC-IDS2017 Cleaned Dataset](https://www.kaggle.com/datasets/dhoogla/cicids2017/data).
2. **Update Path:** Place the downloaded files in a folder named `CICIDS-2017_cleaned` in the root directory.
3. **Execute the Script:** Run the training script:

```bash
python main.py
```
*Two neural networks will be trained and saved in `.onnx` format.*

---

## 🛠 Experimental Workflow

### 4. Logical Representation (MODSAT)
The trained networks must be represented in the **Lukasiewicz infinite-valued logical system**. 
1. Compile and run the **reluka** project.
2. The tool will generate a `.limodsat` file for each output neuron of each neural network.
3. Move these generated files to a folder named `limodsat` in this repository.

### 5. Property Construction
Once the `.limodsat` files are in the correct folder, execute:

```bash
python buildProperties.py
```
This script will generate the logical formulas (security properties) that will be tested by the solver.

### 6. Property Testing (Verification)
Use the **lukasol** solver to verify the properties. From the root directory, run the following command for each property file:

```bash
./bin/Release/lukasol -m 'propertyname'.limodsat
```

---

## 📂 Project Structure
* `main.py`: Training and exporting models.
* `buildProperties.py`: Logic formula generation.
* `/limodsat`: Directory for logical representation files.
* `/docs`: Documentation and thesis reports.