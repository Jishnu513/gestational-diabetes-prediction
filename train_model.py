import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import shap
import joblib

print("=" * 70)
print("GESTATIONAL DIABETES PREDICTION - TRAINING")
print("=" * 70)

# Load data
print("\nðŸ“Š Loading dataset...")
data = pd.read_csv('data/Gestational_Diabetes.csv')
print(f"âœ… Dataset loaded: {data.shape[0]} samples, {data.shape[1]} features")

# Prepare features
features = ['Age', 'Pregnancy No', 'Weight', 'Height', 'BMI', 'Heredity']
X = data[features]
y = data['Prediction']

print(f"\nðŸ“ˆ Class distribution:")
print(f"   No GDM: {(y==0).sum()} ({(y==0).sum()/len(y)*100:.1f}%)")
print(f"   GDM: {(y==1).sum()} ({(y==1).sum()/len(y)*100:.1f}%)")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nðŸ”€ Split: {len(X_train)} train, {len(X_test)} test")

# Create model pipeline
print("\nðŸ—ï¸  Building model pipeline...")
model = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    ))
])

# Train
print("ðŸš€ Training XGBoost model...")
model.fit(X_train, y_train)
print("âœ… Training complete!")

# Evaluate
y_pred = model.predict(X_test)
train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc = accuracy_score(y_test, y_pred)

print(f"\nðŸ“Š Results:")
print(f"   Training Accuracy: {train_acc*100:.2f}%")
print(f"   Testing Accuracy: {test_acc*100:.2f}%")

print(f"\nðŸ“ˆ Classification Report:")
print(classification_report(y_test, y_pred, 
      target_names=['No GDM', 'GDM']))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No GDM', 'GDM'],
            yticklabels=['No GDM', 'GDM'])
plt.title('Confusion Matrix', fontweight='bold')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300)
print("\nðŸ’¾ Saved: confusion_matrix.png")

# SHAP explainability
print("\nðŸ” Computing SHAP values...")
explainer = shap.Explainer(model.named_steps['classifier'], 
                           model.named_steps['scaler'].transform(X_train.sample(100)))
shap_values = explainer(model.named_steps['scaler'].transform(X_test))

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig('shap_summary.png', dpi=300)
print("ðŸ’¾ Saved: shap_summary.png")

# Save model
joblib.dump(model, 'gdm_model.pkl')
print("\nðŸ’¾ Model saved: gdm_model.pkl")

print("\n" + "=" * 70)
print("âœ… TRAINING COMPLETE!")
print("=" * 70)
