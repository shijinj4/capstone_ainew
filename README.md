# ✈️ AI Travel Planner

## AI-Powered Travel Itinerary & Budget Planning System

AI Travel Planner is a web-based travel planning application that helps users create personalized travel plans based on their **destination, trip duration, budget, and travel preferences**.

The application combines **Artificial Intelligence, Machine Learning, and web technologies** to generate travel recommendations while estimating travel-related costs. A machine learning model is used to predict travel expenses and support budget-aware trip planning.

---

## 🌍 Project Overview

Planning a trip can be challenging because travelers need to consider destinations, accommodation, activities, restaurants, transportation, and overall budget.

The AI Travel Planner aims to simplify this process by providing users with an intelligent travel planning experience.

The application allows users to provide information such as:

* Destination
* Number of travel days
* Budget
* Preferred activities
* Travel preferences

The system then uses AI and machine learning to help generate a personalized travel plan and provide estimated travel costs.

---

## 🎯 Project Objectives

The main objectives of this project are:

1. Create personalized travel itineraries using AI.
2. Predict estimated travel costs using Machine Learning.
3. Help users plan trips according to their available budget.
4. Recommend suitable travel activities and experiences.
5. Provide an easy-to-use web interface.
6. Combine AI-generated recommendations with machine-learning-based cost estimation.
7. Provide a foundation for an intelligent travel assistant.

---

## ⭐ Key Features

### 🗺️ AI Travel Planning

Users can enter their travel requirements and receive a personalized travel plan.

### 💰 Budget Prediction

The application uses a trained machine learning model to estimate travel-related costs.

### 📊 Machine Learning

A machine learning model is trained using travel price data and saved as a `.pkl` model for application use.

### 🏨 Budget-Aware Recommendations

The predicted travel cost can be used to help users make better decisions based on their available budget.

### 🖥️ Web Application

The project uses a Python Flask backend with HTML templates and static frontend resources.

### 📱 User-Friendly Interface

The application provides a simple interface for entering travel information and viewing generated results.

---

# 🧠 Artificial Intelligence & Machine Learning

The project contains both AI-based itinerary generation and machine-learning-based price prediction components.

## Machine Learning Component

The repository contains:

```text
budget_prediction.ipynb
budget_model.pkl
budget_prediction_model.pkl
```

The notebook is used for developing the travel budget prediction model, while the trained models are stored as Pickle files for use by the application.

The model can be used to estimate travel costs based on travel-related input features.

### ML Workflow

```text
Travel Dataset
      │
      ▼
Data Preprocessing
      │
      ▼
Feature Selection
      │
      ▼
Model Training
      │
      ▼
Model Evaluation
      │
      ▼
Trained ML Model
      │
      ▼
.pkl Model File
      │
      ▼
Flask Application
      │
      ▼
Predicted Travel Cost
```

---

# 📊 Dataset

The project includes the following dataset:

```text
Expanded_Travel_Price_Prediction_Dataset.csv
```

The dataset contains travel-related information that can be used to train and evaluate the price prediction model.

The dataset supports the machine learning component of the application by providing historical examples for predicting travel costs.

---

# 🏗️ Application Architecture

The application follows a Flask-based web architecture.

```text
                 ┌──────────────────────┐
                 │        User          │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │   Web Interface      │
                 │ HTML / CSS / JS      │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │    Flask Backend     │
                 │       app.py         │
                 └───────┬───────┬──────┘
                         │       │
              ┌──────────┘       └──────────┐
              ▼                             ▼
     ┌─────────────────┐          ┌─────────────────┐
     │ AI Itinerary    │          │ ML Price/Budget │
     │ Generation      │          │ Prediction      │
     └────────┬────────┘          └────────┬────────┘
              │                            │
              └────────────┬───────────────┘
                           ▼
                 ┌──────────────────────┐
                 │ Personalized Travel  │
                 │       Plan           │
                 └──────────────────────┘
```

---

# 🛠️ Technologies Used

## Backend

* Python
* Flask

## Machine Learning

* Python
* Pandas
* NumPy
* Scikit-learn
* Jupyter Notebook
* Pickle

## Frontend

* HTML
* CSS
* JavaScript

## AI

* Generative AI / LLM-based itinerary generation
* AI-assisted travel recommendations

## Development

* Git
* GitHub
* Python Virtual Environment

---

# 📁 Project Structure

```text
capstone_ainew/
│
├── app.py
│   └── Main Flask application
│
├── app_old.py
│   └── Previous version of the Flask application
│
├── budget_prediction.ipynb
│   └── Machine learning model development
│
├── budget_model.pkl
│   └── Trained budget prediction model
│
├── budget_prediction_model.pkl
│   └── Saved travel price prediction model
│
├── Expanded_Travel_Price_Prediction_Dataset.csv
│   └── Travel price prediction dataset
│
├── requirements.txt
│   └── Python dependencies
│
├── templates/
│   └── HTML templates
│
├── static/
│   └── CSS, JavaScript and image resources
│
└── README.md
    └── Project documentation
```

The repository currently contains the `templates` and `static/images` directories along with the Python application, datasets, notebooks, trained models, and dependency file.

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/shijinj4/capstone_ainew.git
```

Navigate to the project:

```bash
cd capstone_ainew
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
```

Activate the environment:

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
```

Activate:

```bash
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Application

After installing the dependencies, run:

```bash
python app.py
```

The Flask development server should start locally.

Open the application in your browser using the local address displayed by Flask.

---

# 🧪 Machine Learning Model

To experiment with the machine learning component:

1. Open:

```text
budget_prediction.ipynb
```

2. Load the travel dataset.

3. Perform data preprocessing.

4. Train the prediction model.

5. Evaluate the model.

6. Save the trained model as a `.pkl` file.

The trained model can then be loaded by the Flask application for travel cost prediction.

---

# 🔄 End-to-End Application Flow

```text
User
 │
 ▼
Enter Destination
 │
 ▼
Enter Trip Duration
 │
 ▼
Enter Budget
 │
 ▼
Select Travel Preferences
 │
 ▼
Flask Application
 │
 ├───────────────┐
 ▼               ▼
AI Itinerary     ML Cost
Generation       Prediction
 │               │
 └───────┬───────┘
         ▼
Budget-Aware Travel Plan
         │
         ▼
Personalized Itinerary
```

---

# 💡 Example Use Case

A user wants to travel for **5 days** with a limited budget.

The user provides:

```text
Destination: Toronto
Duration: 5 days
Budget: $1,000
Activities: Sightseeing, Food, Shopping
```

The application can use the provided information to:

1. Generate a personalized itinerary.
2. Estimate travel-related expenses.
3. Compare the predicted cost with the user's budget.
4. Help the user select suitable activities.
5. Produce a more budget-conscious travel plan.

---

# 🚀 Future Enhancements

The project can be extended with several advanced features.

### 🤖 Advanced AI

* Conversational travel chatbot
* Follow-up questions about the itinerary
* Personalized recommendations
* AI-based destination selection

### 💰 Budget Optimization

* Accommodation price comparison
* Restaurant recommendations based on budget
* Transportation cost estimation
* Daily spending recommendations
* Automatic itinerary adjustment when the budget is exceeded

### 🧠 Machine Learning

* Destination recommendation model
* Hotel recommendation model
* Activity recommendation model
* Travel cost classification
* Multiple ML model comparison
* Automated model selection
* Model performance monitoring

### 🌐 Real-Time Data

Future versions could integrate APIs for:

* Hotel prices
* Flights
* Weather
* Restaurants
* Attractions
* Transportation

This would allow the system to provide more current travel recommendations.

### ☁️ Cloud Deployment

The application could also be deployed using cloud services such as:

* AWS EC2
* AWS Elastic Beanstalk
* AWS Lambda
* Amazon S3
* Amazon RDS
* Amazon SageMaker

---

# 🔐 Security Considerations

For production deployment:

* Store API keys in environment variables.
* Never commit API keys or credentials to GitHub.
* Use HTTPS.
* Validate user input.
* Apply authentication where required.
* Protect external API credentials.
* Use secure database credentials.
* Restrict access to production services.

---

# 🎓 Academic Project

This project was developed as an academic capstone project to demonstrate the integration of:

**Artificial Intelligence + Machine Learning + Web Development**

The project demonstrates how AI-generated travel recommendations can be combined with machine-learning-based cost prediction to create a more intelligent travel planning experience.

---

# 👨‍💻 Author

**Shijin Joseph**

GitHub:

https://github.com/shijinj4

---

# 📄 License

This project is intended primarily for academic and educational purposes.
