# Graph Data Mining for Anti-Money Laundering: Detecting Illicit Transactions in the Bitcoin Blockchain (Elliptic Dataset) 

**MA384 Data Mining Project: Graph Analysis of the Elliptic Dataset**  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)  

**Collaborators:** *Ian Lemons, Aidan O'Neil, Ethan Pabbathi, Rhys Phelps*

**Advised by:** *Dr. Yosi Shibberu*


## Overview  
This repository contains our **Data Mining (MA384)** project to identify illicit patterns in Bitcoin transactions using the [Elliptic Dataset](https://www.kaggle.com/ellipticco/elliptic-data-set). Our goal is to apply graph mining, feature engineering, and machine learning techniques to detect suspicious behavior on the blockchain.

---

## Repository Structure  
```
.
├── notebooks/           # Jupyter notebooks for EDA, modeling, and analysis
│   ├── exploratory/     # Initial data exploration and visualization
│   └── modeling/        # Model training and evaluation
├── data/                # Dataset references and preprocessing scripts
├── scripts/             # Reusable Python modules (e.g., graph utilities)
├── docs/                # Meeting notes, research papers, and documentation
├── reports/             # Generated reports and visualizations (PDFs, charts)
├── .gitignore           # Ignores large files, outputs, and secrets
├── requirements.txt     # Python dependencies
├── LICENSE              # MIT License
└── README.md            # This file
```
---

## Dataset  
The [Elliptic Dataset](https://www.kaggle.com/ellipticco/elliptic-data-set) contains:  
- **Nodes**: 203,769 Bitcoin transactions (21% illicit, 2% licit, 77% unknown).  
- **Edges**: 234,355 directed payment flows.  
- **Features**: 166 anonymized transaction features.  
---

## License  
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.  

---

## Acknowledgments  
- Dataset: [Elliptic](https://www.elliptic.co/)  
- Course: MA384 Data Mining, [RHIT](https://www.rose-hulman.edu) 
- Advisor: Dr. Yosi Shiberru  

---

