from utils.preprocessing import load_data, scale_data
from models.dnn import build_dnn
import numpy as np

# load + train
X_train, X_test, y_train, y_test = load_data()
X_train, X_test = scale_data(X_train, X_test)

model = build_dnn(X_train.shape[1])
model.fit(X_train, y_train, epochs=10, verbose=0)

# save
model.save("model.h5")

# also save stats
np.save("mean.npy", np.mean(X_train, axis=0))
np.save("std.npy", np.std(X_train, axis=0))

print("Model saved successfully")