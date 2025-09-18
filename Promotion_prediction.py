import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load dataset
df = pd.read_csv("Data.csv")  
print(df.head())

# 2. Features & Target
X = df.drop("Promotion", axis=1)
y = df["Promotion"]

# Identify categorical and numeric columns
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

# Column transformer (OneHot for categorical, Scaling for numeric)
ct = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", StandardScaler(), numeric_cols),
    ]
)

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Model training
model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
X_train = ct.fit_transform(X_train)
X_test = ct.transform(X_test)
model.fit(X_train, y_train)

# 5. Evaluation
y_pred = model.predict(X_test)
print("\nModel trained successfully")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 6. User Input Prediction
print("\n--- Employee Promotion Prediction ---")
age = int(input("Enter Age: "))
dept = input("Enter Department (Sales/HR/Tech/etc): ")
edu = input("Enter Education (Bachelors/Masters/PhD/etc): ")
exp = int(input("Enter Years of Experience: "))
prev_prom = int(input("Enter Previous Promotions (count): "))
trainings = int(input("Enter Trainings Attended: "))

# Prepare user input
user_df = pd.DataFrame([{
    "Age": age,
    "Department": dept,
    "Education": edu,
    "Experience": exp,
    "Previous_Promotions": prev_prom,
    "Trainings_Attended": trainings
}])

user_X = ct.transform(user_df)
prediction = model.predict(user_X)[0]

print("\nPrediction:", "Promoted" if prediction == 1 else "Not Promoted")
