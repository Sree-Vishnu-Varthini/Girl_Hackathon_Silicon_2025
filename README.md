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

### **6. Integration & Automation**  
- Designed for **fast execution** for automated timing checks.  
- Runs on raw RTL files without requiring full synthesis, reducing **design iteration time**.  

This approach ensures **fast and accurate** combinational depth estimation and **early-stage timing verification**, significantly improving design efficiency.


## Proof of Correctness

## Dataset

## Installation & Setup

## Complexity Analysis

## Alternatives Considered

## Future Work

## References

## Contributors

## Acknowledgements

## Contact Informantion
