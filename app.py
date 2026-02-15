import streamlit as st
import numpy as np
import pickle

# ----------------------------------
# Page config
# ----------------------------------
st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="🧑‍💻",
    layout="wide"  # Changed to wide for better input layout
)

st.markdown("""
    <style>
    /* Make sidebar wider and prevent overlap */
    section[data-testid="stSidebar"] {
        width: 400px !important;
    }
    section[data-testid="stSidebar"] > div {
        width: 400px !important;
    }
    
    /* Sidebar font sizes */
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        font-size: 28px;
    }
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] li {
        font-size: 20px;
    }
    </style>
""", unsafe_allow_html=True)

st.sidebar.title("AICW (Artificial Intelligence Career For Women) By Microsoft and SAP in collaboration with Edunet Foundation")
st.sidebar.title("👥 Team Information")
st.sidebar.write("**Team Members:**")
st.sidebar.write("- Chakra Lakshmi Duppalapudi")
st.sidebar.write("- Poornima Chintala")
st.sidebar.write("- Padmasri Talari")
st.sidebar.write("- Sirisha Alla")
st.sidebar.title("PROJECT GUIDE")
st.sidebar.write("### Abdul Aziz Md")
st.sidebar.write("Master Trainer - Edunet Foundation")

st.title("🧑‍💻 Employee Attrition Prediction")
st.write("Enter employee details below to predict attrition risk.")

# ----------------------------------
# Load model and scaler
# ----------------------------------
@st.cache_resource
def load_artifacts():
    with open("new_Rf_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_artifacts()

# ----------------------------------
# Feature inputs (organized in columns for better UX)
# ----------------------------------
Mapping = {
    'Gender': {'Male': 1, 'Female': 0},
    'Department': {'Finance': 0, 'HR': 1, 'IT': 2, 'Marketing': 3, 'Sales': 4},
    'Job_Role': {'Analyst': 0, 'Assistant': 1, 'Executive': 2, 'Manager': 3},
    'Overtime': {'Yes': 1, 'No': 0}
}

col1, col2 = st.columns(2)
with col1:
    age = st.number_input('Age', min_value=20, max_value=80, value=30)
    gender = st.radio('Gender', options=['Male', 'Female'], horizontal=True)
    department = st.selectbox('Department', options=['Sales', 'IT', 'HR', 'Marketing', 'Finance'])
    job_role = st.segmented_control('Job_Role', options=['Manager', 'Assistant', 'Executive', 'Analyst'])
    job_level = st.number_input('Job_Level', min_value=1, max_value=5, value=2)
    monthly_income = st.slider('Monthly_Income', min_value=5000, max_value=150000, value=50000)

with col2:
    hourly_rate = st.number_input('Hourly_Rate', min_value=10, max_value=100, value=50)
    years_at_company = st.number_input('Years_at_Company', min_value=1, max_value=30, value=5)
    years_in_current_role = st.number_input('Years_in_Current_Role', min_value=1, max_value=20, value=3)
    overtime = st.radio('Overtime', options=['Yes', 'No'], horizontal=True)
    work_life_balance = st.number_input('Work_Life_Balance', min_value=1, max_value=5)
    job_satisfaction = st.number_input('Job_Satisfaction', min_value=1, max_value=5)

col3, col4 = st.columns(2)
with col3:
    performance_rating = st.number_input('Performance_Rating', min_value=1, max_value=5)
    training_hours = st.number_input('Training_Hours_Last_Year', min_value=0, max_value=100, value=20)
    project_count = st.number_input('Project_Count', min_value=1, max_value=10, value=4)
    absenteeism = st.number_input('Absenteeism', min_value=0, max_value=50, value=5)

with col4:
    work_env_satisfaction = st.number_input('Work_Environment_Satisfaction', min_value=1, max_value=4)
    distance_from_home = st.number_input('Distance_From_Home', min_value=1, max_value=50, value=10)
    num_companies_worked = st.number_input('Number_of_Companies_Worked', min_value=0, max_value=9, value=2)


#Feature order must match training data
feature_names = [
    'Age', 'Gender', 'Department', 'Job_Role', 'Job_Level', 'Monthly_Income',
    'Hourly_Rate', 'Years_at_Company', 'Years_in_Current_Role', 'Work_Life_Balance',
    'Job_Satisfaction', 'Performance_Rating', 'Training_Hours_Last_Year',
    'Overtime', 'Project_Count', 'Absenteeism', 'Work_Environment_Satisfaction',
    'Distance_From_Home', 'Number_of_Companies_Worked'
]
input_values = [age, gender, department, job_role, job_level, monthly_income,
                hourly_rate, years_at_company, years_in_current_role, work_life_balance,
                job_satisfaction, performance_rating, training_hours, overtime,
                project_count, absenteeism, work_env_satisfaction, distance_from_home,
                num_companies_worked]

# ----------------------------------
# Prediction
# ----------------------------------
if st.button("🧑‍💻 Predict Attrition", type="primary"):
    try:
        # Map categoricals
        processed_values = []
        for i, fname in enumerate(feature_names):
            val = input_values[i]
            if fname in Mapping:
                mapped = Mapping[fname].get(val, val)
                processed_values.append(mapped)
            else:
                processed_values.append(val)

        input_array = np.array(processed_values).reshape(1, -1)
        scaled_input = scaler.transform(input_array)
        prediction = model.predict(scaled_input)[0]
        prob = model.predict_proba(scaled_input)[0]

        attrition_result = "Yes (High Risk)" if prediction == 1 else "No (Low Risk)"
        st.success(f"**Prediction: {attrition_result}**")
        st.info(f"**Probability of Attrition: {prob[1]*100:.1f}%**")
        
        # Optional: Show feature importance or confidence
        st.metric("Attrition Confidence", f"{max(prob)*100:.1f}%")
    except Exception as e:
        st.error(f"Prediction error: {str(e)}. Check model files and feature order.")
