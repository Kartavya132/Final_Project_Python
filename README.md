# Titanic Dataset — Exploratory Data Analysis (EDA)

A structured, comprehensive data analysis project designed to inspect, clean, and visualize the Titanic dataset. This project explores key factors influencing passenger survival rates, such as socio-economic status (class), gender, age, fare, family size on board, and embarkation port using Python, Pandas, Matplotlib, and Seaborn.

Both a standalone Python script (`Final_project.py`) and a Jupyter Notebook (`Titanic.ipynb`) are provided for flexible execution.

---

## 📁 Repository Structure

```text
├── database/
│   ├── train.csv                
│   ├── test.csv                 
│   └── gender_submission.csv    
├── plots/All Plots
├── Final_project.py             
├── Titanic.ipynb                
└── README.md                    

```

---

## 🛠️ Installation & Setup

### Prerequisites

Ensure you have Python 3.8+ installed along with the required data science libraries:

```bash
pip install pandas matplotlib seaborn

```

### Running the Analysis

* **Via Python Script:**
```bash
python Final_project.py

```


* **Via Jupyter Notebook:**
Open `Titanic.ipynb` in your preferred IDE (VS Code, Jupyter Lab, etc.) and execute the cells sequentially.

---

## 🔍 Analysis Pipeline & Expected Outcomes

The project executes a comprehensive end-to-end exploratory data analysis pipeline. Below are the specific operational steps and outcomes generated during runtime.

### 1. Data Inspection & Console Summaries

Upon execution, the pipeline handles initial data ingestion and outputs critical structural telemetry directly to the console:

* **Dataset Shapes:** Dynamically tracks dimensions of both `train.csv` and `test.csv`.
* **Structural Integrity (`.info()`):** Examines column data types, memory footprint, and evaluates non-null counts.
* **Missing Value Identification:** Scans the dataset for missing data (`NaN` values) across features like `Age`, `Cabin`, and `Embarked`.
* **Statistical Descriptive Profiling:** Generates central tendencies, standard deviations, and boundaries for numeric and categorical variables using `describe(include="all")`.
* **Overall Survival Baseline:** Computes and prints the global historical survival rate percentage of the training pool:
> **Overall Survival Rate:** ~38.38%



### 2. Data Cleaning & Feature Engineering

Before rendering visualizations, the pipeline applies targeted data transformations:

* **Age Imputation:** Fills missing `Age` values using the **median** age of the passenger collective to avoid skewing distribution characteristics.
* **Embarked Imputation:** Replaces missing categorical `Embarked` gaps with the **mode** (most frequent port of embarkation).
* **Feature Generation (`FamilySize`):** engineers a holistic engineering metric calculating total family presence on board:

$$\text{FamilySize} = \text{SibSp} + \text{Parch} + 1$$



---

## 📊 Visualizations Generated

The project leverages customized configurations (high-DPI, explicit styling parameters, hidden spines) to output 9 high-quality visualizations depicting survival dependencies:

| Visualization | Description | Key Insight Addressed |
| --- | --- | --- |
| **Survival Count** | A categorical bar plot splitting aggregate volumes of passengers who survived vs. those who perished. | Visualizes the baseline imbalance of the core target label. |
| **Survival Rate by Gender** | A bar chart evaluating the average survival probability metrics between Male and Female demographics. | Assesses the "women and children first" historical maritime protocol. |
| **Survival Rate by Passenger Class** | An analysis evaluating survival rates across Socio-economic status tiers (`Pclass` 1, 2, and 3). | Identifies survival probability correlations relative to ticket pricing tiers. |
| **Age Distribution by Survival Status** | A detailed histogram complete with Kernel Density Estimate (KDE) overlays tracking age cohorts grouped by survival. | Highlights vulnerable age categories (e.g., infants and the elderly) vs higher mortality age groups. |
| **Overall Age Distribution** | A unified histogram detailing passenger volume frequency distributions across the lifespan. | Identifies the predominant age demographic profiles present on the vessel. |
| **Fare Distribution** | A distribution histogram mapping skewness profiles across passenger ticket purchase prices. | Highlights major financial skewness, showing low-cost ticket majorities vs high-premium anomalies. |
| **Survival Rate by Embarkation Port** | A metric-focused bar plot parsing survival likelihood across ports: Southampton (**S**), Cherbourg (**C**), and Queenstown (**Q**). | Investigates whether passenger geographical boarding locations correlate with survival rates. |
| **Survival Rate by Family Size** | A breakdown plotting survival likelihood averages against discrete family counts on board. | Explores the balance between traveling solo, with nuclear families, or with large family cohorts. |
| **Correlation Heatmap** | A visual Pearson product-moment correlation matrix tracking relationships among all numeric data features. | Quantifies cross-variable dependencies and detects multicollinearity issues. |

---

## 📈 Sample Plot Layout Standards

All plots generated conform to modern visualization standards:

* **High Resolution:** Configured at `120 DPI` for crisp presentation output.
* **Clean Geometry:** Top and right chart borders (spines) are automatically hidden to maximize data-to-ink ratio.
* **Automated Annotations:** Categorical charts utilize dynamic layout container algorithms (`ax.bar_label`) to imprint absolute values or percentages cleanly over data bars.