House Price Prediction Project

Overview

This project is a machine learning model designed to predict house prices based on various features such as location, size, number of bedrooms, bathrooms, and total square footage. The dataset used for this project is from Kaggle, and the implementation leverages Python and several key libraries including Pandas, NumPy, Matplotlib, and Scikit-learn.

Steps Followed 

1. Data Import and Exploration

Imported the necessary libraries: pandas, numpy, matplotlib, and scikit-learn.

Loaded the dataset using Pandas:

df = pd.read_csv("Bengaluru_House_Data.csv")

Performed an initial exploration to understand the structure, shape, and contents of the dataset using methods like .head(), .info(), and .describe().

2. Data Cleaning

Removed unnecessary columns such as area_type, society, balcony, and availability.

Dropped rows with missing values using:

df = df.dropna()

Processed the size column to extract the number of bedrooms (BHK).

Converted non-numeric values in total_sqft to numeric values using a custom function to handle ranges and invalid entries.

3. Feature Engineering

Created a new column price_per_sqft by dividing the price by the total square footage.

Consolidated less common locations into a single category called other to reduce noise in the data.

Removed outliers based on:

Price per square foot.

BHK-specific price patterns.

Bathrooms exceeding the number of bedrooms by more than 2.

Used one-hot encoding for the location column to convert categorical data into numerical data.

4. Model Preparation

Split the data into features (X) and target variable (y):

X = df.drop("price", axis=1)
y = df["price"]

Performed a train-test split with a test size of 20%:

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

5. Model Training and Evaluation

Trained multiple regression models including:

Linear Regression

Lasso Regression

Decision Tree Regressor

Used GridSearchCV to identify the best hyperparameters for each model.

Evaluated models using cross-validation to ensure stability and generalization.

6. Prediction Function

Created a custom prediction function:

def predict_price(location, sqft, bath, bhk):
    loc_index = np.where(X.columns == location)[0]
    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if len(loc_index) > 0:
        x[loc_index[0]] = 1
    return model.predict([x])[0]

Used the function to predict house prices for given inputs.

Libraries Used

Pandas: Data manipulation and cleaning.

NumPy: Numerical computations.

Matplotlib: Data visualization.

Scikit-learn: Model training, evaluation, and feature processing.

Results

Linear Regression: Achieved an accuracy of 86%.

Random Forest Regression: Achieved an accuracy of 70%.

Lasso Regression and Decision Tree: Tuned using GridSearchCV to find the optimal parameters.

Final model was chosen based on cross-validation scores and prediction performance on the test set.

How to Use

Clone the repository and install the necessary dependencies:

git clone <repo_link>
cd house-price-prediction
pip install -r requirements.txt

Add the dataset Bengaluru_House_Data.csv to the project directory.

Run the Python script to train the model and make predictions:

python house_price_prediction.py

Use the predict_price function to estimate house prices based on input parameters like location, square footage, number of bathrooms, and bedrooms.

Future Enhancements

Add more features such as proximity to schools, hospitals, and transportation.

Use advanced machine learning algorithms like Gradient Boosting (XGBoost, LightGBM).

Incorporate external data sources to enrich feature engineering.

Build a web-based interface to make the model user-friendly.

Credits

Dataset: Kaggle

Libraries: Scikit-learn, Pandas, NumPy, Matplotlib

