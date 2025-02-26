# ML-Based Combinational Depth and Timing Violation Predictor

## Overview
Timing analysis is a critical step in chip design, ensuring that signals meet required constraints. The traditional approach relies on post-synthesis reports, which are computationally expensive and time-consuming.

This project proposes an AI-driven approach to predict combinational logic depth for given RTL signals before synthesis, allowing for early timing violation detection. We leverage machine learning models trained on RTL design data to estimate logic depth efficiently. This method significantly reduces design iteration cycles, enhances debugging speed, and improves overall project execution time.

## Problem Statement
Timing analysis is essential in digital circuit design to ensure signals meet required constraints. Timing violations occur when a signal's combinational logic depth exceeds the allowable limit for a given clock cycle. Traditional post-synthesis analysis is slow, often taking hours or days for large designs, delaying debugging and optimization.

This project introduces an AI-driven approach to predict combinational logic depth in RTL modules before synthesis. By training machine learning models on RTL design data, we provide fast, accurate estimates, significantly reducing timing validation time.

Designed for hardware engineers and chip design companies, our solution enables early detection of potential timing violations, minimizing design iterations, accelerating development, and improving efficiency in semiconductor design.

## Features
✅ Predicts combinational depth of signals directly from RTL code, eliminating the need for full synthesis. <br>
✅ Identifies setup and hold time violations based on gate delays and clock frequency. <br>
✅ Extracts circuit structure by parsing RTL and constructing a directed acyclic graph (DAG) representation.<br>
✅ Uses machine learning models (Support Vector Regressor, Random Forest, Linear Regression) for depth estimation.<br>
✅ Automates early-stage timing verification, reducing design iteration time.<br>
✅ Supports custom gate delay values, allowing flexibility for different technologies.<br>
✅ Fast execution compared to traditional synthesis-based timing analysis.<br>
✅ Scalable and adaptable for different RTL designs and clock constraints.<br>

## **Methodology**  

Our approach leverages **machine learning** and **graph-based analysis** to predict **combinational depth** and identify **timing violations** without requiring full synthesis.  

### **1. Dataset Creation**  
- Collected RTL implementations and extracted equations representing combinational logic.  
- Labeled data with combinational depths.  

### **2. Feature Engineering**  
- Parsed RTL to extract gate-level features by counting occurrences of **AND, OR, NOT, NAND, NOR, XOR, XNOR, and BUFFER** gates.  
- Constructed a **directed acyclic graph (DAG)** representing circuit structure:  
  - **Nodes** represent signals.  
  - **Edges** represent dependencies between logic gates.  
- Computed **combinational depth** as the **longest path** in the graph.  

### **3. Combinational Depth Calculation**  
- Used the constructed DAG to determine the **longest path** from inputs to the target signal.  
- This depth serves as the ground truth for ML model training.  

### **4. Machine Learning Model Training**  
- Trained three models for combinational depth prediction:  
  - **Support Vector Regressor (SVR)** – Best accuracy.  
  - **Random Forest Regressor** – Handles non-linearity well.  
  - **Linear Regression** – Simple baseline model.  
- **Input features:**  
  - Gate counts (AND, OR, NOT, etc.).  
  - Combinational depth.  
  - Final output signal.  
- **Target variable:**  
  - **Combinational delay** (used to determine timing violations).  

### **5. Timing Violation Analysis**  
- Computed **min and max delays** using predefined gate delays:  
  - **Min delay** = Sum of minimum delays along the longest path.  
  - **Max delay** = Sum of maximum delays along the longest path.  
- Detected timing violations by checking:  
  - **Setup Violation:** `Max Delay ≥ Clock Period`.  
  - **Hold Violation:** `Min Delay ≤ 10% of Clock Period`.  

### **6. Model Evaluation & Validation**  
- **Dataset split:** **80% training, 20% testing**.  
- **Performance metrics used:**  
  - **Mean Absolute Error (MAE)**  
  - **Mean Squared Error (MSE)**  
  - **R² Score**  

### **7. Integration & Automation**  
- Designed for **fast execution** for automated timing checks.  
- Runs on raw RTL files without requiring full synthesis, reducing **design iteration time**.  

This approach ensures **fast and accurate** combinational depth estimation and **early-stage timing verification**, significantly improving design efficiency.


## Proof of Correctness

## **Dataset**  

The dataset was **self-created** to ensure high-quality, labeled data for training and validating the machine learning model. It consists of RTL implementations, extracted gate-level features, and combinational depth values derived from synthesis reports.  

### **1. Data Collection**  
- RTL modules were manually designed and analyzed.  
- Logic equations were extracted to represent **combinational circuits**.  

### **2. Feature Set**  
Each RTL module was processed to extract the following key features:  
- **Gate Counts:** Number of AND, OR, NOT, NAND, NOR, XOR, XNOR, and BUFFER gates.  
- **Combinational Depth:** Longest path in the **directed acyclic graph (DAG)** representation of the circuit.  
- **Final Output Signal:** Captures logic dependency structure.  

### **3. Labeling & Ground Truth**  
- **Combinational Depth** serves as the target variable for ML training.  
- Depth values were manually verified to ensure accuracy.  

### **4. Data Validation**  
- Extracted gate counts and circuit structures were manually reviewed.  
- DAG-based combinational depth calculations were cross-checked.  

This dataset enables the **ML model to predict combinational depth accurately and detect setup/hold violations efficiently**, improving early-stage timing analysis.  

## **Installation & Setup**  

Follow these steps to set up the environment and run the project from scratch.  

### **Step 1: Install Anaconda (Recommended)**  
Anaconda provides a pre-configured Python environment, making setup easier.  

- Download and install Anaconda from:  
  [https://www.anaconda.com/products/distribution](https://www.anaconda.com/products/distribution)  <br>
  
- Open **Anaconda Prompt** and create a new environment:  
  ```sh
  conda create --name timing_analysis python=3.9
  ```
- Activate the environment:  
  ```sh
  conda activate timing_analysis
  ```

---

### **Step 2: Clone the Repository**  
Download the project files from GitHub:  
```sh
git clone https://github.com/your-repo-link
cd your-repo
```

---

### **Step 3: Install Dependencies**  

Install the dependencies:  
```sh
pip install numpy pandas scikit-learn networkx regex
```

---

### **Step 4: Running the Code**  
To analyze an RTL module and predict combinational depth, run:  
```sh
python main.py --rtl data/example_rtl.v --clock 10
```
Replace `"data/example_rtl.v"` with your RTL file path and adjust the clock period as needed.  

---

### **Step 5: Using Jupyter Notebook (Optional)**  
For interactive development and debugging:  
```sh
conda install jupyter  
jupyter notebook
```
Open the notebook and run the provided scripts to visualize circuit graphs and ML predictions.  

---

### **Step 6: Verifying Installation**  
Run the following test to check if everything is working correctly:  
```sh
python -c "import numpy, pandas, sklearn, networkx; print('Setup Successful!')"
```

---

## **Complexity Analysis**  

The computational complexity of the algorithm is analyzed based on its key components:  

### **1. Feature Extraction (Graph Construction)**  
- **Process:** Parse RTL to extract gate types and connections, then construct a **directed acyclic graph (DAG)**.  
- **Time Complexity:** \(O(V + E)\), where **V** is the number of signals and **E** is the number of connections (edges).  

### **2. Combinational Depth Calculation**  
- **Process:** Find the longest path in the DAG to determine combinational depth.  
- **Time Complexity:**  
  - Using **Topological Sorting + DAG Longest Path Algorithm:** \(O(V + E)\).  
  - Worst-case scenario (fully connected graph): \(O(V^2)\).  

### **3. Machine Learning Model Training**  
- **Support Vector Regressor (SVR):**  
  - **Training:** \(O(N^2)\) (where \(N\) is the dataset size).  
  - **Prediction:** \(O(N)\).  
- **Random Forest Regressor:**  
  - **Training:** \(O(N log N)\).  
  - **Prediction:** \(O( log N)\).  
- **Linear Regression:**  
  - **Training:** \(O(N d^2)\) (where \(d\) is the number of features).  
  - **Prediction:** \(O(d)\).  

## **Alternatives Considered**  

During the development of this project, several alternative approaches were explored for predicting combinational depth and detecting timing violations. Below are the key alternatives, along with their advantages and reasons for rejection.  

### **1. Graph Neural Networks (GNNs)**  
**Description:**  
GNNs model circuit structures effectively by learning dependencies between gates in a graph representation.  

**Advantages:**  
✅ Captures complex relationships between signals and gates.  
✅ Scales well for large circuits.  
✅ Reduces the need for manual feature engineering.  

**Reasons for Rejection:**  
❌ Requires **large datasets** for effective training, which were not readily available.  
❌ **High computational cost**, making it impractical for quick timing analysis.  
❌ **Difficult interpretability**, making debugging and validation harder.  

### **2. Rule-Based Heuristics**  
**Description:**  
A predefined set of rules calculates combinational depth based on gate counts and structural patterns.  

**Advantages:**  
✅ **Fast execution**, as it runs in constant time.  
✅ No need for training, making it easy to implement.  

**Reasons for Rejection:**  
❌ **Low generalization**, as heuristics may not cover complex circuit variations.  
❌ **Limited accuracy**, especially in handling reconvergent paths and varying gate delays.  
❌ Requires **manual updates** when new gate types or optimizations are introduced.  

### **3. Traditional Graph Algorithms (Longest Path Calculation Without ML)**  
**Description:**  
Using **longest path algorithms** on a directed acyclic graph (DAG) to determine combinational depth directly.  

**Advantages:**  
✅ **Provides exact combinational depth** without needing ML.  
✅ No dataset or model training required.  

**Reasons for Rejection:**  
❌ **Does not account for gate delays**, assuming all gates contribute equally to delay.  
❌ **Scalability concerns**, as large circuits can lead to expensive computations.  
❌ **Lack of adaptability**, since the approach does not improve with more data.  

### **4. Static Timing Analysis (STA) with EDA Tools**  
**Description:**  
Integrating early-stage Static Timing Analysis (STA) tools instead of using ML-based predictions.  

**Advantages:**  
✅ **High accuracy**, as STA is a well-established industry standard.  
✅ Can factor in **placement and routing effects** for realistic timing predictions.  

**Reasons for Rejection:**  
❌ **Computationally expensive**, making it slower than an ML-based approach.  
❌ **Tool dependency**, requiring proprietary EDA tools that may not be accessible in all environments.  
❌ Defeats the purpose of **early-stage lightweight timing prediction**.  

### **Final Choice: Machine Learning-Based Prediction**  
After evaluating alternatives, an **ML-based approach** was chosen as it offers the best balance of **speed, accuracy, and adaptability**. The **Support Vector Regressor (SVR), Random Forest, and Linear Regression** models allow for:  
- **Quick combinational depth estimation** without full synthesis.  
- **Improved adaptability** by learning from circuit patterns.  
- **Efficient runtime**, making it suitable for early-stage timing analysis.  

This approach ensures **fast and accurate predictions**, enabling early detection of potential timing violations before synthesis. 🚀  

## **Future Work**  

While this project successfully predicts combinational depth and detects timing violations, several areas can be improved and expanded in future iterations.  

### **1. Enhancing Model Accuracy**  
- Explore **deep learning models**, such as Graph Neural Networks (GNNs), to improve prediction accuracy.  
- Incorporate additional features like **fan-in, fan-out, and wire delays** to refine timing estimations.  
- Train on **larger datasets** with real-world circuits for better generalization.  

### **2. Optimizing Performance**  
- Implement **parallel processing** techniques to speed up feature extraction and graph traversal.  
- Explore lightweight **approximate algorithms** for combinational depth estimation to further reduce runtime.  

### **3. Integrating with EDA Tools**  
- Develop a **plugin for popular EDA tools** (e.g., Synopsys, Cadence) for seamless timing prediction integration.  
- Enable real-time feedback for designers during **RTL development** without running full synthesis.  

### **4. Expanding Functionality**  
- Extend the tool to support **sequential circuits**, predicting depth across pipeline stages.  
- Include **power and area estimation**, allowing for multi-objective optimization beyond timing analysis.  
- Support different **technology nodes** (e.g., 5nm, 7nm) by incorporating **process-specific gate delays**.  

By implementing these future improvements, this project can become an even more powerful tool for **early-stage timing analysis and optimization** in digital design.


## References
[1] Baños, Raul & Gil, Consolación & Montoya, Maria & Ortega, Julio. (2003). A Parallel Evolutionary Algorithm for Circuit Partitioning.. 365-371. 10.1109/EMPDP.2003.1183612. <br> <br> 
[2] Bairamkulov, R., Friedman, E. (2023). Graphs in VLSI circuits and systems. In: Graphs in VLSI. Springer, Cham. https://doi.org/10.1007/978-3-031-11047-4_3<br> <br> 
[3] Wang, Jichao. (2024). A hybrid deep learning and clonal selection algorithm-based model for commercial building energy consumption prediction. Science Progress. 107. 10.1177/00368504241283360. <br> <br> 
[4] Olabiyi, Winner. (2024). APPLYING MACHINE LEARNING FOR TIMING ANALYSIS IN DIGITAL CIRCUITS USING PYRTL. <br>

## Contact Informantion
- Sree Vishnu Varthini S, Pre-Final Year Student, Sri Eshwar College of Engineering, sreevishnuvarthini@gmail.com
