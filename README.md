# 🧠 Non-Intrusive Cognitive Load Estimation System

## Features
- Web-based Python IDE with real-time code execution
- Behavioral data collection (typing speed, pauses, errors)
- Advanced feature extraction from code & behavior
- XGBoost ML model for 3-class cognitive load prediction
- Real-time analytics dashboard
- Research-grade model evaluation metrics

## Quick Start

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

1. **Install dependencies**
```bash
pip install -r requirements.txt

2. **Start Live IDE Server**

python app.py

🚀 Server running: http://localhost:5000
Live IDE ready!

3. **Create Output Directories**

mkdir data
mkdir evaluation_plots

4. Run Research Evaluation

python research_evaluation.py

Files saved:
- research_report.txt
- evaluation_plots/confusion_matrices.png
- evaluation_plots/feature_importance.png

That's all......

----------------------short implementation line by line:

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

1.pip install -r requirements.txt
2.python app.py

###### For the output.

3.mkdir data
4.mkdir evaluation_plots
5.python research_evaluation.py

###### For the research and accuracy.