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

## Setup  
### 1. Clone the Repository  
```bash
git clone https://github.com/rhit-oneilat/datamining-aml-elliptic-detection.git
cd datamining-aml-elliptic-detection
```

### 2. Create a Virtual Environment  
#### Using `venv`:  
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

#### Using `conda`:  
```bash
conda create -n bitcoin-aml python=3.9
conda activate bitcoin-aml
pip install -r requirements.txt
```

### 3. Install Kaggle API (for Dataset Access)  
1. Follow the [Kaggle API setup guide](https://github.com/Kaggle/kaggle-api) to add your `kaggle.json` credentials.  
2. Download the dataset:  
```bash
kaggle datasets download -d ellipticco/elliptic-data-set
unzip elliptic-data-set.zip -d data/raw/
```

---

## Dataset  
The [Elliptic Dataset](https://www.kaggle.com/ellipticco/elliptic-data-set) contains:  
- **Nodes**: 203,769 Bitcoin transactions (21% illicit, 2% licit, 77% unknown).  
- **Edges**: 234,355 directed payment flows.  
- **Features**: 166 anonymized transaction features.  

---

## Project Tracking  
We use **GitHub Projects** for task management:  
- [Project Board](https://github.com/your-username/bitcoin-aml-elliptic-datamining/projects/1)  
- **Milestones**:  
  1. EDA & Visualization (Due: YYYY-MM-DD)  
  2. Feature Engineering (Due: YYYY-MM-DD)  
  3. Model Training (Due: YYYY-MM-DD)  
  4. Final Report (Due: YYYY-MM-DD)  

### Weekly Tasks  
- Assign tasks via GitHub Issues (label: `task`).  
- Track progress in `docs/meeting_notes/`.  

---

## Contributing  
1. **Branch Naming**: `[type]/[short-description]` (e.g., `feature/graph-embeddings`, `bugfix/data-load`).  
2. **Commit Messages**: Use [Conventional Commits](https://www.conventionalcommits.org/):  
   ```bash
   git commit -m "feat: add Louvain clustering script"
   git commit -m "fix: resolve edge list parsing bug"
   ```
3. **Pull Requests**:  
   - Link PRs to issues (e.g., `Closes #12`).  
   - Request reviews from at least 2 teammates.  


---

## Documentation  
- **Research Notes**: See `docs/research/` for summaries of relevant papers.  
- **Technical Writeups**: Use `docs/methods/` to document algorithms (e.g., PageRank, GNNs).  
- **Final Report**: Draft in `reports/final/` (LaTeX or Markdown).  

---

## License  
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.  

---

## Acknowledgments  
- Dataset: [Elliptic](https://www.elliptic.co/)  
- Course: MA384 Data Mining, [RHIT](https://www.rose-hulman.edu) 
- Advisor: Dr. Yosi Shiberru  

---

