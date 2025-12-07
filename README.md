import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Create 2000 records
num_records = 2000

# Generate unique Patient IDs
patient_ids = [f'PAT{str(i).zfill(6)}' for i in range(100001, 102001)]

# Enhanced domain definitions with Country and Region
genders = ['Male', 'Female']
blood_types = ['A+', 'A-', 'B+', 'B-', 'O+', 'O-', 'AB+', 'AB-']

# Define countries and their regions
countries = ['USA', 'Canada', 'UK', 'Australia', 'Germany', 'France', 'Japan', 'India']
country_regions = {
    'USA': ['Northeast', 'Midwest', 'South', 'West', 'Pacific'],
    'Canada': ['Eastern Canada', 'Central Canada', 'Western Canada', 'Northern Canada'],
    'UK': ['England', 'Scotland', 'Wales', 'Northern Ireland'],
    'Australia': ['Eastern Australia', 'Western Australia', 'Northern Territory', 'Southern Australia'],
    'Germany': ['Northern Germany', 'Southern Germany', 'Eastern Germany', 'Western Germany'],
    'France': ['Île-de-France', 'Northern France', 'Southern France', 'Eastern France', 'Western France'],
    'Japan': ['Hokkaido', 'Tohoku', 'Kanto', 'Chubu', 'Kansai', 'Chugoku', 'Shikoku', 'Kyushu'],
    'India': ['Northern India', 'Southern India', 'Eastern India', 'Western India', 'Central India']
}

admission_types = ['Emergency', 'Elective', 'Urgent', 'Transfer']
medical_conditions = [
    'Diabetes', 'Hypertension', 'Asthma', 'Arthritis', 'Heart Disease',
    'Cancer', 'Stroke', 'COPD', 'Kidney Disease', 'Liver Disease',
    'Pneumonia', 'Sepsis', 'Fracture', 'Appendicitis', 'Gallstones',
    'COVID-19', 'Influenza', 'Bronchitis', 'Migraine', 'Anemia'
]
medications = [
    'Insulin', 'Metformin', 'Lisinopril', 'Atorvastatin', 'Levothyroxine',
    'Amlodipine', 'Metoprolol', 'Albuterol', 'Omeprazole', 'Losartan',
    'Simvastatin', 'Hydrochlorothiazide', 'Prednisone', 'Gabapentin', 'Furosemide',
    'Warfarin', 'Ibuprofen', 'Acetaminophen', 'Azithromycin', 'Ciprofloxacin'
]
insurance_providers = [
    'UnitedHealth', 'Anthem', 'Aetna', 'Cigna', 'Humana',
    'Blue Cross', 'Kaiser', 'Molina', 'Centene', 'HCSC'
]
test_results = ['Normal', 'Abnormal', 'Critical', 'Pending']
hospitals = [
    'General Hospital', 'City Medical Center', 'University Hospital',
    'Community Hospital', 'Memorial Hospital', 'Regional Medical Center',
    'Childrens Hospital', 'Veterans Hospital'
]
doctor_names = [
    'Dr. Smith', 'Dr. Johnson', 'Dr. Williams', 'Dr. Brown', 'Dr. Jones',
    'Dr. Miller', 'Dr. Davis', 'Dr. Garcia', 'Dr. Rodriguez', 'Dr. Wilson',
    'Dr. Martinez', 'Dr. Anderson', 'Dr. Taylor', 'Dr. Thomas', 'Dr. Moore',
    'Dr. Lee', 'Dr. Harris', 'Dr. Clark', 'Dr. Lewis', 'Dr. Robinson'
]
departments = [
    'Cardiology', 'Emergency', 'Orthopedics', 'Oncology', 'Neurology',
    'Pediatrics', 'General Surgery', 'Internal Medicine', 'Radiology', 'ICU',
    'Maternity', 'Psychiatry', 'Dermatology', 'ENT', 'Urology'
]
discharge_statuses = ['Recovered', 'Transferred', 'Against Medical Advice', 'Deceased', 'Home Care']
procedure_types = ['Surgery', 'Diagnostic Test', 'Therapy', 'Medication', 'Observation', 'Rehabilitation']
lab_tests = ['Blood Test', 'X-Ray', 'MRI', 'CT Scan', 'Ultrasound', 'EKG', 'Urinalysis', 'Biopsy']
risk_levels = ['Low', 'Medium', 'High', 'Critical']
readmission_status = ['No', 'Yes - 7 days', 'Yes - 30 days', 'Yes - 90 days']
patient_satisfaction = ['Very Satisfied', 'Satisfied', 'Neutral', 'Dissatisfied', 'Very Dissatisfied']
complication_types = ['None', 'Infection', 'Bleeding', 'Allergic Reaction', 'Surgical Complication']
payment_methods = ['Insurance', 'Cash', 'Credit Card', 'Payment Plan', 'Government']
occupation_types = ['Professional', 'Laborer', 'Retired', 'Student', 'Unemployed', 'Healthcare Worker']
marital_statuses = ['Single', 'Married', 'Divorced', 'Widowed']
smoking_status = ['Never', 'Former', 'Current']
alcohol_consumption = ['None', 'Occasional', 'Regular']
allergies = ['None', 'Penicillin', 'Latex', 'NSAIDs', 'Food', 'Other']
follow_up_required = ['Yes', 'No', 'Pending']

# Generate base dates (last 5 years)
start_date = datetime(2020, 1, 1)
end_date = datetime(2024, 12, 31)

# Generate admission dates
admission_dates = []
for _ in range(num_records):
    days_diff = (end_date - start_date).days
    random_days = random.randint(0, days_diff)
    admission_date = start_date + timedelta(days=random_days)
    admission_dates.append(admission_date)

# Generate length of stay with more realistic distribution
length_of_stay = np.clip(np.random.exponential(scale=5, size=num_records), 1, 45).astype(int)
# Make some stays longer for serious conditions
serious_conditions = np.random.choice([0, 1], num_records, p=[0.7, 0.3])
length_of_stay = np.where(serious_conditions == 1, 
                         length_of_stay + np.random.randint(5, 20, num_records), 
                         length_of_stay)

# Generate discharge dates based on admission date and length of stay
discharge_dates = []
for i in range(num_records):
    discharge_date = admission_dates[i] + timedelta(days=int(length_of_stay[i]))
    discharge_dates.append(discharge_date)

# Calculate readmission probability (higher for certain conditions)
readmission_prob = np.random.random(num_records)
readmission_flag = np.where(readmission_prob > 0.85, 
                           np.random.choice(['Yes - 7 days', 'Yes - 30 days', 'Yes - 90 days'], num_records), 
                           'No')

# Generate BMI (Body Mass Index)
bmi = np.round(np.random.normal(26, 5, num_records), 1)
bmi = np.clip(bmi, 16, 45)

# Generate vital signs
systolic_bp = np.random.normal(125, 15, num_records).astype(int)
systolic_bp = np.clip(systolic_bp, 90, 200)
diastolic_bp = np.random.normal(80, 10, num_records).astype(int)
diastolic_bp = np.clip(diastolic_bp, 60, 120)
heart_rate = np.random.normal(75, 10, num_records).astype(int)
heart_rate = np.clip(heart_rate, 50, 130)
temperature = np.round(np.random.normal(37, 0.5, num_records), 1)
temperature = np.clip(temperature, 35.5, 40.5)

# Generate lab values
glucose = np.random.normal(100, 25, num_records).astype(int)
glucose = np.clip(glucose, 60, 300)
cholesterol = np.random.normal(190, 30, num_records).astype(int)
cholesterol = np.clip(cholesterol, 120, 300)

# Generate costs breakdown
medication_cost = np.round(length_of_stay * np.random.uniform(50, 200, num_records), 2)
lab_cost = np.round(np.random.exponential(500, num_records), 2)
procedure_cost = np.round(np.random.exponential(1500, num_records), 2)
room_cost = np.round(length_of_stay * np.random.uniform(800, 1500, num_records), 2)

# Total billing amount
billing_amount = medication_cost + lab_cost + procedure_cost + room_cost
billing_amount = np.round(np.clip(billing_amount, 500, 100000), 2)

# Generate Country and Region data
country_list = []
region_list = []

# Create distribution (more patients from some countries)
country_distribution = np.random.choice(countries, num_records, 
                                       p=[0.35, 0.15, 0.12, 0.08, 0.10, 0.08, 0.07, 0.05])

for country in country_distribution:
    country_list.append(country)
    # Select a region within the country
    available_regions = country_regions[country]
    region_list.append(np.random.choice(available_regions))

# Create the enhanced dataset with Country and Region
data = {
    # Existing columns (1-17)
    'Patient_ID': patient_ids,
    'Patient_Age': np.random.randint(18, 95, num_records),
    'Patient_Gender': np.random.choice(genders, num_records),
    'Blood_Type': np.random.choice(blood_types, num_records),
    'Medical_Condition': np.random.choice(medical_conditions, num_records),
    'Admission_Type': np.random.choice(admission_types, num_records),
    'Test_Results': np.random.choice(test_results, num_records, p=[0.40, 0.35, 0.15, 0.10]),
    'Medication': np.random.choice(medications, num_records),
    'Hospital_Name': np.random.choice(hospitals, num_records),
    'Department': np.random.choice(departments, num_records),
    'Doctor_Name': np.random.choice(doctor_names, num_records),
    'Admission_Date': [date.strftime('%Y-%m-%d') for date in admission_dates],
    'Discharge_Date': [date.strftime('%Y-%m-%d') for date in discharge_dates],
    'Length_of_Stay_Days': length_of_stay,
    'Billing_Amount': billing_amount,
    'Insurance_Provider': np.random.choice(insurance_providers, num_records),
    'Discharge_Status': np.random.choice(discharge_statuses, num_records, p=[0.75, 0.10, 0.05, 0.05, 0.05]),
    
    # NEW: Country and Region columns
    'Country': country_list,  # Country column
    'Region': region_list,    # Region column
    
    # Additional columns (20-44)
    'Procedure_Type': np.random.choice(procedure_types, num_records),
    'Lab_Test_Performed': np.random.choice(lab_tests, num_records),
    'Patient_Risk_Level': np.random.choice(risk_levels, num_records, p=[0.5, 0.3, 0.15, 0.05]),
    'Readmission_Status': readmission_flag,
    'Patient_Satisfaction': np.random.choice(patient_satisfaction, num_records, p=[0.3, 0.4, 0.2, 0.08, 0.02]),
    'Complications': np.random.choice(complication_types, num_records, p=[0.85, 0.05, 0.04, 0.03, 0.03]),
    'Payment_Method': np.random.choice(payment_methods, num_records, p=[0.7, 0.1, 0.1, 0.1]),
    'Patient_Occupation': np.random.choice(occupation_types, num_records),
    'Marital_Status': np.random.choice(marital_statuses, num_records, p=[0.3, 0.5, 0.15, 0.05]),
    'Smoking_Status': np.random.choice(smoking_status, num_records, p=[0.5, 0.3, 0.2]),
    'Alcohol_Consumption': np.random.choice(alcohol_consumption, num_records, p=[0.4, 0.4, 0.2]),
    'Known_Allergies': np.random.choice(allergies, num_records, p=[0.7, 0.1, 0.05, 0.05, 0.05, 0.05]),
    'Follow_Up_Required': np.random.choice(follow_up_required, num_records, p=[0.6, 0.3, 0.1]),
    
    # Clinical Metrics
    'BMI': bmi,
    'Systolic_BP': systolic_bp,
    'Diastolic_BP': diastolic_bp,
    'Heart_Rate': heart_rate,
    'Temperature_C': temperature,
    'Glucose_mg_dL': glucose,
    'Cholesterol_mg_dL': cholesterol,
    
    # Cost Breakdown
    'Medication_Cost': medication_cost,
    'Lab_Cost': lab_cost,
    'Procedure_Cost': procedure_cost,
    'Room_Cost': room_cost,
    
    # Calculated fields
    'Cost_per_Day': np.round(billing_amount / length_of_stay, 2),
    'Insurance_Coverage_Percent': np.random.randint(70, 101, num_records)  # 70-100% coverage
}

# Create DataFrame
df = pd.DataFrame(data)

# Reorder columns for better organization (Country and Region placed strategically)
column_order = [
    # Patient Identification & Demographics
    'Patient_ID', 'Patient_Age', 'Patient_Gender', 'Blood_Type', 
    'Country', 'Region',  # NEW: Country and Region added here
    'Marital_Status', 'Patient_Occupation', 'Smoking_Status', 
    'Alcohol_Consumption', 'Known_Allergies', 'BMI',
    
    # Medical Information
    'Medical_Condition', 'Admission_Type', 'Test_Results', 'Medication',
    'Procedure_Type', 'Lab_Test_Performed', 'Patient_Risk_Level',
    'Complications', 'Follow_Up_Required',
    
    # Clinical Metrics
    'Systolic_BP', 'Diastolic_BP', 'Heart_Rate', 'Temperature_C',
    'Glucose_mg_dL', 'Cholesterol_mg_dL',
    
    # Hospital Administration
    'Hospital_Name', 'Department', 'Doctor_Name',
    'Admission_Date', 'Discharge_Date', 'Length_of_Stay_Days',
    'Discharge_Status', 'Readmission_Status', 'Patient_Satisfaction',
    
    # Financial Information
    'Billing_Amount', 'Medication_Cost', 'Lab_Cost', 
    'Procedure_Cost', 'Room_Cost', 'Cost_per_Day',
    'Insurance_Provider', 'Insurance_Coverage_Percent', 'Payment_Method'
]

df = df[column_order]

# Save to CSV
output_file = 'healthcare_dataset_with_country_region_47_columns.csv'
df.to_csv(output_file, index=False)

# Calculate comprehensive KPIs with Country/Region analysis
total_patients = len(df)
total_billing = df['Billing_Amount'].sum()
avg_billing = df['Billing_Amount'].mean()
avg_length_of_stay = df['Length_of_Stay_Days'].mean()
abnormal_tests = df[df['Test_Results'].isin(['Abnormal', 'Critical'])].shape[0]
readmission_rate = df[df['Readmission_Status'] != 'No'].shape[0] / total_patients * 100
satisfaction_rate = df[df['Patient_Satisfaction'].isin(['Very Satisfied', 'Satisfied'])].shape[0] / total_patients * 100
complication_rate = df[df['Complications'] != 'None'].shape[0] / total_patients * 100
avg_bmi = df['BMI'].mean()

# Country-specific analysis
country_stats = df.groupby('Country').agg({
    'Patient_ID': 'count',
    'Billing_Amount': ['sum', 'mean'],
    'Length_of_Stay_Days': 'mean',
    'Patient_Satisfaction': lambda x: (x.isin(['Very Satisfied', 'Satisfied'])).mean() * 100,
    'Readmission_Status': lambda x: (x != 'No').mean() * 100
}).round(2)

country_stats.columns = ['Patient_Count', 'Total_Billing', 'Avg_Billing', 
                         'Avg_Length_of_Stay', 'Satisfaction_Rate', 'Readmission_Rate']

# Region analysis within top countries
top_countries = df['Country'].value_counts().head(3).index.tolist()
region_analysis = {}
for country in top_countries:
    country_data = df[df['Country'] == country]
    region_stats = country_data['Region'].value_counts()
    region_analysis[country] = region_stats

# Create comprehensive summary
print("=" * 100)
print("HEALTHCARE DATASET WITH COUNTRY & REGION COLUMNS (47 COLUMNS TOTAL)")
print("=" * 100)
print(f"📁 File created: {output_file}")
print(f"📊 Total Records: {total_patients:,}")
print(f"📈 Total Columns: {len(df.columns)}")

print("\n🌍 COUNTRY AND REGION INFORMATION ADDED:")
print(f"  • Countries included: {', '.join(countries)}")
print(f"  • Total countries: {len(df['Country'].unique())}")
print(f"  • Total regions: {len(df['Region'].unique())}")

print("\n📋 DATASET STRUCTURE (47 columns):")
print("  CORE DEMOGRAPHICS & GEOGRAPHY (13 columns):")
geo_cols = column_order[:13]
for i, col in enumerate(geo_cols, 1):
    if col in ['Country', 'Region']:
        print(f"    {i:2d}. {col} 🌍")  # Flag geography columns
    else:
        print(f"    {i:2d}. {col}")

print("\n  MEDICAL INFORMATION (9 columns):")
med_cols = column_order[13:22]
for i, col in enumerate(med_cols, 1):
    print(f"    {i+13:2d}. {col}")

print("\n  CLINICAL METRICS (6 columns):")
clinical_cols = column_order[22:28]
for i, col in enumerate(clinical_cols, 1):
    print(f"    {i+22:2d}. {col}")

print("\n  HOSPITAL ADMINISTRATION (9 columns):")
admin_cols = column_order[28:37]
for i, col in enumerate(admin_cols, 1):
    print(f"    {i+28:2d}. {col}")

print("\n  FINANCIAL INFORMATION (10 columns):")
finance_cols = column_order[37:]
for i, col in enumerate(finance_cols, 1):
    print(f"    {i+37:2d}. {col}")

print("\n📊 COUNTRY-LEVEL KPIs:")
print("\n  Patient Distribution by Country:")
country_dist = df['Country'].value_counts()
for country, count in country_dist.items():
    percentage = (count / total_patients) * 100
    print(f"    • {country:<10}: {count:>4} patients ({percentage:.1f}%)")

print("\n  Performance Metrics by Country:")
print(country_stats.to_string())

print("\n🌐 REGION ANALYSIS (Top 3 Countries):")
for country in top_countries:
    print(f"\n  {country}:")
    regions = region_analysis[country]
    for region, count in regions.head(3).items():
        percentage = (count / country_dist[country]) * 100
        print(f"    • {region:<25}: {count:>3} patients ({percentage:.1f}%)")

print("\n📈 NEW KPIs ENABLED BY COUNTRY/REGION DATA:")
print("  1. Geographic Performance Analysis")
print("  2. Regional Disease Prevalence")
print("  3. Country-wise Cost Comparisons")
print("  4. Regional Healthcare Access Metrics")
print("  5. Cross-border Patient Flow Analysis")
print("  6. Insurance Coverage by Country")
print("  7. Satisfaction Rates by Region")
print("  8. Readmission Patterns by Geography")

print("\n💡 SUGGESTED GEOGRAPHIC VISUALIZATIONS FOR POWER BI:")
print("  1. World Map: Patient distribution by country")
print("  2. Drill Down Maps: Country → Region → Hospital")
print("  3. Comparative Analysis: Metrics across countries")
print("  4. Heat Maps: Disease prevalence by region")
print("  5. Bubble Charts: Cost vs. Satisfaction by country")

print("\n🔍 SAMPLE DATA WITH GEOGRAPHY (first 3 rows):")
sample_columns = ['Patient_ID', 'Patient_Age', 'Country', 'Region', 
                  'Medical_Condition', 'Admission_Type', 'Length_of_Stay_Days', 
                  'Billing_Amount', 'Hospital_Name']
print(df[sample_columns].head(3).to_string(index=False))

print("\n📊 OVERALL KPIs:")
print(f"  • Total Billing Amount:           ${total_billing:,.2f}")
print(f"  • Average Billing Amount:         ${avg_billing:,.2f}")
print(f"  • Average Length of Stay:         {avg_length_of_stay:.2f} days")
print(f"  • Abnormal/Critical Tests:        {abnormal_tests/total_patients*100:.1f}%")
print(f"  • Patient Satisfaction Rate:      {satisfaction_rate:.1f}%")
print(f"  • Readmission Rate:               {readmission_rate:.1f}%")

print("\n✅ DATA VALIDATION:")
print(f"  • All countries have valid regions: {df.groupby('Country')['Region'].nunique().min() >= 1}")
print(f"  • Geographic data complete: {df[['Country', 'Region']].isnull().sum().sum() == 0}")
print(f"  • Date consistency: {pd.to_datetime(df['Discharge_Date']) >= pd.to_datetime(df['Admission_Date']).all()}")

print("=" * 100)
