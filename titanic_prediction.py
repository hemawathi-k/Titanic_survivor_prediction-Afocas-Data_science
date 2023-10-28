import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load the training and testing datasets
train_data = pd.read_csv('train (2).csv')
test_data = pd.read_csv('test (2).csv')

# Step 2: Data Preprocessing

# Handling missing values in the training data
train_data['Age'].fillna(train_data['Age'].mean(), inplace=True)
train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace=True)
train_data['Sex'] = train_data['Sex'].map({'male': 0, 'female': 1})
train_data = pd.get_dummies(train_data, columns=['Embarked'])

# Handling missing values in the testing data
test_data['Age'].fillna(test_data['Age'].mean(), inplace=True)
test_data['Embarked'].fillna(test_data['Embarked'].mode()[0], inplace=True)
test_data['Sex'] = test_data['Sex'].map({'male': 0, 'female': 1})
test_data = pd.get_dummies(test_data, columns=['Embarked'])

# Remove non-essential columns
non_essential_columns = ['Name', 'Ticket', 'Cabin']
train_data.drop(columns=non_essential_columns, inplace=True)
test_data.drop(columns=non_essential_columns, inplace=True)

# Splitting the training data into features (X) and target variable (y)
X_train = train_data.drop('Survived', axis=1)
y_train = train_data['Survived']

# Step 3: Exploratory Data Analysis (EDA)

# Relationship between passenger classes and survival using a bar plot
plt.figure(figsize=(8, 6))
sns.barplot(x='Pclass', y='Survived', data=train_data)
plt.title('Survival Rate by Passenger Class')
plt.show()

# Relationship between gender and survival using a bar plot
plt.figure(figsize=(8, 6))
sns.barplot(x='Sex', y='Survived', data=train_data)
plt.title('Survival Rate by Gender')
plt.show()

# Step 4: Analyze the visualized data for insights

# Analyzing Passenger Class vs. Survival
class_survival = train_data[['Pclass', 'Survived']].groupby('Pclass').mean()
print("Survival Rate by Passenger Class:")
print(class_survival)

# Analyzing Gender vs. Survival
gender_survival = train_data[['Sex', 'Survived']].groupby('Sex').mean()
print("\nSurvival Rate by Gender:")
print(gender_survival)

# Step 5: Utilize a classification algorithm for prediction

# Split the training set for model training and evaluation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Initialize the Logistic Regression model
model = LogisticRegression(max_iter=1000)

# Fit the model on the training data
model.fit(X_train, y_train)

# Make predictions on the validation set
y_pred = model.predict(X_val)

# Calculate accuracy and show the classification report
accuracy = accuracy_score(y_val, y_pred)
print("\nAccuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_val, y_pred))
