import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

# بيانات تدريب بسيطة (تجريبية)
X = np.array([
    [7, 3, 20, 10],
    [5, 8, 100, 60],
    [6, 5, 50, 30],
    [4, 10, 150, 80],
    [8, 2, 10, 5],
    [3, 11, 180, 120]
])

# 0 = Low, 1 = Medium, 2 = High
y = np.array([0, 2, 1, 2, 0, 2])

# إنشاء النموذج
model = RandomForestClassifier()

# تدريب النموذج
model.fit(X, y)

# حفظ النموذج
joblib.dump(model, "rf_model.pkl")

print("Model saved successfully!")