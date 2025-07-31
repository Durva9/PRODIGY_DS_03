# PRODIGY_DS_03
📊 Bank Marketing Prediction Dashboard

This project presents an interactive Streamlit dashboard powered by a Decision Tree Classifier to predict whether a customer will subscribe to a term deposit. Using the Bank Marketing dataset, it uncovers key behavioral and demographic patterns that influence purchase decisions in marketing campaigns.

---

🔍 Key Objectives

- Predict customer subscription based on personal and campaign-related features.
- Analyze the influence of key variables such as job, marital status, education, and previous outcomes.
- Visualize top features contributing to customer decisions.
- Provide an easy-to-use interface for exploring model performance and insights.

---

📌 Key Features

📄 **Dataset Preview**: Displays a sample of the Bank Marketing dataset directly in the app.

📈 **Model Accuracy**: Computes and displays model accuracy and a detailed classification report.

🧠 **Feature Importance**: Interactive Plotly bar chart of the top 10 influential features in predicting outcomes.

🧪 **Machine Learning Model**: Decision Tree Classifier trained on a one-hot encoded version of the dataset with a depth limit of 5.

🎨 **Streamlit UI**: Responsive and user-friendly layout, expandable sections, and embedded visualizations.

---

🚀 Tools & Technologies

- **📊 Streamlit**: For building the web app interface
- **📦 Scikit-learn**: For training the Decision Tree Classifier and evaluating performance
- **📈 Plotly**: For generating interactive feature importance charts
- **🧮 Pandas**: For data manipulation and preprocessing

---

📂 Dataset

- **Name**: Bank Marketing Dataset  
- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/bank+marketing)  
- **Records**: ~45,000  
- **Target Variable**: `y` (Customer subscribed: yes/no)  
- **Format**: CSV (`;` separated)  

---

📈 Insights Derived

- Customers with previous campaign contact history are more likely to subscribe.
- Job type, duration of call, and previous outcome are strong predictors.
- The Decision Tree Classifier achieves solid accuracy with balanced feature importance.
- Visualization of model insights enables stakeholders to make informed marketing decisions.
