from utils.preprocessing import load_data, scale_data
from models.baseline import train_lr, train_rf
from models.dnn import build_dnn
from utils.metrics import evaluate
from imblearn.over_sampling import SMOTE
from models.gan import train_gan, generate_samples
import numpy as np

print("STARTING PROJECT...\n")

# ===== LOAD DATA =====
X_train, X_test, y_train, y_test = load_data()
print("Data loaded")

# ===== SMOTE =====
sm = SMOTE()
X_train, y_train = sm.fit_resample(X_train, y_train)
print("SMOTE applied")

# ===== SCALING =====
X_train, X_test = scale_data(X_train, X_test)
print("Scaling done")


# =========================
# 🔥 GAN AUGMENTATION
# =========================
print("\n--- GAN Augmentation ---")

# Extract minority class (cancer = 1)
X_minority = X_train[y_train == 1]

print("Minority samples:", len(X_minority))
print("Majority samples:", len(X_train[y_train == 0]))

# Train GAN
generator = train_gan(X_minority)

# Generate synthetic samples
n_samples = len(X_train[y_train == 0]) - len(X_minority)
n_samples = max(1, n_samples)  # safety fix

print("Generating samples:", n_samples)

synthetic_samples = generate_samples(generator, n_samples)

# Combine GAN data
X_train_gan = np.vstack([X_train, synthetic_samples])
y_train_gan = np.hstack([y_train, np.ones(n_samples)])

print("GAN data generated and balanced")


# =========================
# 📊 BASELINE MODELS
# =========================
print("\n--- Logistic Regression ---")
lr = train_lr(X_train_gan, y_train_gan)
pred = lr.predict(X_test)
prob = lr.predict_proba(X_test)[:, 1]
evaluate(y_test, pred, prob)


print("\n--- Random Forest ---")
rf = train_rf(X_train_gan, y_train_gan)
pred = rf.predict(X_test)
prob = rf.predict_proba(X_test)[:, 1]
evaluate(y_test, pred, prob)


# =========================
# 🧠 DEEP NEURAL NETWORK
# =========================
print("\n--- Deep Neural Network ---")

dnn = build_dnn(X_train_gan.shape[1])

dnn.fit(
    X_train_gan,
    y_train_gan,
    epochs=10,
    batch_size=32,
    verbose=1,
    class_weight={0: 1, 1: 3}
)

prob = dnn.predict(X_test).flatten()
pred = (prob > 0.5).astype(int)

evaluate(y_test, pred, prob)


print("\nPROJECT COMPLETED")