# 📊 Customer Intelligence System  
### Churn Prediction + Customer Segmentation + Business Personas

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Machine Learning](https://img.shields.io/badge/ML-Churn%20Prediction-green)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-yellow)
![Clustering](https://img.shields.io/badge/KMeans-Segmentation-purple)

---

## 🧠 Overview

This project is an end-to-end customer analytics system that combines **machine learning and clustering** to transform raw telecom customer data into actionable business insights.

It identifies:
- Customers likely to churn  
- Customer behavior segments  
- High-value at-risk users  
- Business personas for targeted marketing  

---

## 🧱 Architecture

Raw Customer Data  
→ Data Cleaning & Preprocessing  
→ Churn Prediction Model (Random Forest)  
→ Customer Segmentation (K-Means)  
→ Business Persona Mapping  
→ Final Insight Dataset Export  

---

## 🎯 Business Problem

Companies lose revenue when customers leave unexpectedly.

This system helps answer:
- Who is likely to churn?
- Why are they leaving?
- What types of customers do we have?
- What actions should the business take?

---

## ⚙️ Workflow

1. Data cleaning and preprocessing  
2. Feature engineering  
3. Churn prediction using Random Forest  
4. Customer segmentation using K-Means clustering  
5. Business persona creation  
6. Export of final analytics dataset  

---

## 📊 Output Example

`customer_insights_output.csv`

| Customer ID | Tenure | Monthly Charges | Churn Probability | Segment |
|--------------|--------|----------------|------------------|----------|
| 7590-VHVEG   | 2      | 85.5           | 0.87             | 2        |

---

## 🧠 Key Insights

- Identifies high-risk high-value customers for retention campaigns  
- Groups customers into behavioral segments  
- Enables proactive marketing and personalization strategies  
- Improves customer lifetime value (CLV) targeting  

---

## 🧠 Model Components

### 🔹 Churn Prediction
- Algorithm: Random Forest Classifier  
- Output: Probability of customer churn  

### 🔹 Customer Segmentation
- Algorithm: K-Means Clustering  
- Output: Customer behavior groups  

---

## 🧠 Business Personas

- 🔴 High-risk high-value customers → retention focus  
- 🟡 New users → onboarding optimization  
- 🟢 Loyal customers → upsell opportunities  
- 🔵 Average users → general marketing  

---

## 🛠️ Tech Stack

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib  

---

## 📦 How to Run

```bash
pip install pandas scikit-learn
python customer_intelligence.py
