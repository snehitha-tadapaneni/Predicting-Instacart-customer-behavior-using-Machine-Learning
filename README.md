# 🛍️ Customer Purchase Prediction using Machine Learning

This project builds an end-to-end **Machine Learning pipeline** to predict the likelihood of a customer making a purchase based on their browsing, demographic, and interaction data.  
It demonstrates the complete ML lifecycle from **data cleaning** and **feature engineering** to **hyperparameter tuning** and **model evaluation**tailored for real-world e-commerce analytics.

---

## 📋 Overview
E-commerce companies thrive on understanding **customer intent** knowing who is likely to buy, when, and what.  
This project applies **supervised learning** techniques to identify key behavioral signals that influence customer purchases.  
By combining **robust preprocessing**, **model optimization**, and **performance evaluation**, it delivers an interpretable, scalable pipeline that can be integrated into retail analytics workflows.

---

## ⚙️ Environment Setup
This notebook is designed to be executed in Google Colab.
Before running, ensure the following:

- Mount your Google Drive.

- Include the utilities folder containing p2_shallow_learning_utilities.py.

- Include the models folder containing p2_shallow_learning_models.py.

### 💡 Note: Avoid running this in remote systems; Colab provides the best compatibility for file paths and dependencies.

---

## 📂 Dataset Access

The dataset used in this project is too large to be hosted directly on GitHub.  
You can access it from the following link:

🔗 [Download Dataset from Google Drive]([https://www.kaggle.com/datasets/psparks/instacart-market-basket-analysis])

Once downloaded, upload the dataset to your Google Drive and place it under: Data Folder

---

## 🎯 Objectives
- Clean and preprocess raw user and transaction datasets.  
- Perform **feature engineering** to extract purchase-related indicators.  
- Train and compare multiple **classification models** to predict purchase intent.  
- Optimize hyperparameters for peak performance and interpretability.  
- Generate insights into **top features driving customer conversion**.

---

## 🧠 Machine Learning Models
- Logistic Regression  
- Random Forest Classifier  
- Gradient Boosting  
- Support Vector Machines  
- Neural Networks (TensorFlow / Keras)

---

## ⚙️ Technologies Used
| Category | Tools |
|-----------|-------|
| Programming | Python, Google Colab |
| Data Handling | Pandas, NumPy |
| Modeling | Scikit-Learn, TensorFlow |
| Visualization | Matplotlib, Seaborn |
| Optimization | GridSearchCV, RandomizedSearchCV |

---

## 📈 Results & Insights
- Built a predictive model achieving **high accuracy and precision** in identifying potential buyers.  
- Discovered **key behavioral drivers** such as session duration, product views, and prior purchase frequency.  
- Created an **automated ML pipeline** ready for deployment in customer segmentation or marketing personalization systems.

---

## 💡 Business Impact
- Enables **targeted marketing campaigns** by identifying high-intent users.  
- Helps reduce ad spend through **optimized customer segmentation**.  
- Enhances user experience by supporting **personalized recommendations** and promotions.  
- Supports strategic planning through **data-driven sales forecasting**.

---

## 🧩 Project Structure
(the code.ipynb) file is organized as follows:
code.ipynb
- Data Preparation
- Data Preprocessing
- Feature Engineering
- Model Training
- Hyperparameter Tuning
- Evaluation & Insights


---

## 🚀 Future Enhancements
- Integration with live user data streams for **real-time predictions**.  
- Deployment via **Flask/FastAPI** for production inference in progress.  
- Incorporation of **deep learning architectures** and **feature selection** techniques for further improvement.



