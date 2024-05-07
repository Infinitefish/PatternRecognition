# Generating 3D make-moons data

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def make_moons_3d(n_samples=500, noise=0.1):
    # Generate the original 2D make_moons data
    t = np.linspace(0, 2 * np.pi, n_samples)
    x = 1.5 * np.cos(t)
    y = np.sin(t)
    z = np.sin(2 * t)  # Adding a sinusoidal variation in the third dimension

    # Concatenating the positive and negative moons with an offset and noise
    X = np.vstack([np.column_stack([x, y, z]), np.column_stack([-x, y - 1, -z])])
    y = np.hstack([np.zeros(n_samples), np.ones(n_samples)])

    # Adding Gaussian noise
    X += np.random.normal(scale=noise, size=X.shape)

    return X, y

# Generate the data (1000 datapoints)
X, labels = make_moons_3d(n_samples=1000, noise=0.2)
X_test, labels_test = make_moons_3d(n_samples=500, noise=0.3)
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X, labels)
Y_pred = model.predict(X_test)
accuracy = accuracy_score(labels_test, Y_pred)
print("Accuracy: {}".format(accuracy))
# Plotting
fig = plt.figure()
ax1 = fig.add_subplot(211)
scatter1 = ax1.scatter(X_test[:, 0], X_test[:, 1], c=labels_test, cmap='viridis', marker='o')
legend1 = ax1.legend(*scatter1.legend_elements(), title="Classes")
ax1.add_artist(legend1)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_title("precise classification")
ax2 = fig.add_subplot(212)
scatter2 = ax2.scatter(X_test[:, 0], X_test[:, 1], c=Y_pred, cmap='viridis', marker='o')
legend2 = ax2.legend(*scatter2.legend_elements(), title="Classes")
ax2.add_artist(legend2)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_title("predicted classification")
plt.show()