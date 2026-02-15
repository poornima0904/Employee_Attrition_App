🎯 Project Overview
Built to help HR teams identify employees at risk of leaving by analyzing 19 key factors including job satisfaction, work-life balance, overtime, and compensation. Achieves high prediction accuracy through ensemble learning.

Key Features:

Real-time predictions via web interface

19 feature inputs matching enterprise HR datasets

Pre-trained Random Forest model with scaler

Clean, professional UI with team credits

🧪 Live Demo
Enter employee details (age, department, satisfaction scores, etc.)

Click "Predict Attrition"

Get instant result with probability score

📁 Tech Stack
text
Frontend: Streamlit
Backend: scikit-learn Random Forest
Data Processing: pandas, numpy, pickle
Deployment: Streamlit Cloud / GitHub
🚀 Quick Start
Clone the repo

bash
git clone https://github.com/poornima0904/Employee-Attrition-App.git
cd Employee-Attrition-App
Install dependencies

bash
pip install -r requirements.txt
Run locally

bash
streamlit run app.py
Required files: new_Rf_model.pkl, scaler.pkl

📊 Features Analyzed
Category	                   Features
Demographics	               Age, Gender, Distance from Home
Job	                         Department, Job Role, Job Level, Years at Company
Compensation	               Monthly Income, Hourly Rate
Performance	                 Job Satisfaction, Performance Rating, Training Hours
Workload	                   Overtime, Project Count, Absenteeism
Environment	                 Work-Life Balance, Work Environment Satisfaction
👥 Team
AICW Program - Artificial Intelligence Career For Women
Microsoft + SAP + Edunet Foundation

Team Members:

Chakra Lakshmi Duppalapudi

Poornima Chintala

Padmasri Talari

Sirisha Alla

Project Guide: Abdul Aziz Md
Master Trainer - Edunet Foundation

📈 Model Performance
Algorithm: Random Forest Classifier

Preprocessing: StandardScaler normalization

Features: 19 HR variables

Output: Binary classification (Attrition: Yes/No)

🔮 Future Enhancements
 SHAP explanations for feature importance

 Multi-model comparison (XGBoost, Logistic Regression)

 Historical trend analysis

 Export predictions to CSV

 Azure ML integration

📄 License
MIT License - free to use and modify.

Built with ❤️ for empowering women in AI 🚀
