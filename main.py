import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('data/credit_risk_dataset.csv')

print("Shakl:", df.shape)
print("\nBirinchi 5 qator:")
print(df.head())
# =====================
# 2. MA'LUMOTNI O'RGANISH
# =====================
print("\n=== ASOSIY MA'LUMOTLAR ===")
print(f"Jami mijozlar: {len(df):,}")
print(f"Default bo'lgan: {df['loan_status'].sum():,}")
print(f"Default foizi: {df['loan_status'].mean()*100:.1f}%")
print("\n=== BO'SH QIYMATLAR ===")
print(df.isnull().sum())

# =====================
# 3. TOZALASH
# =====================
df_clean = df.copy()
df_clean['loan_int_rate'].fillna(df_clean['loan_int_rate'].median(), inplace=True)
df_clean['person_emp_length'].fillna(0, inplace=True)
df_clean = df_clean[df_clean['person_age'] < 100]

le = LabelEncoder()
for col in ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']:
    df_clean[col] = le.fit_transform(df_clean[col])

print("\nTozalandi! Shakl:", df_clean.shape)

# =====================
# 4. MODEL
# =====================
X = df_clean.drop('loan_status', axis=1)
y = df_clean['loan_status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_prob = rf.predict_proba(X_test)[:, 1]

print("\n=== RANDOM FOREST NATIJASI ===")
print(classification_report(y_test, rf_pred, target_names=['Yaxshi', 'Default']))
print(f"ROC-AUC: {roc_auc_score(y_test, rf_prob):.4f}")