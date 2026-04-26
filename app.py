from models.dnn import build_dnn
from utils.preprocessing import load_data, scale_data
import numpy as np

# Load data
X_train, X_test, y_train, y_test = load_data()
X_train, X_test = scale_data(X_train, X_test)

print("Data loaded")

# Build model
print("App features:", len(feature_names))

print("Training model...")
model.fit(X_train, y_train, epochs=10, verbose=1)

# ✅ SAVE WEIGHTS
model.save_weights("model.weights.h5")

print("Weights saved")

# Save stats
np.save("mean.npy", np.mean(X_train, axis=0))
np.save("std.npy", np.std(X_train, axis=0))

print("All files saved successfully")
