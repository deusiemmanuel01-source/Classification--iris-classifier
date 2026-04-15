import streamlit as st
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

# ---------------------------
# LOAD DATA
# ---------------------------
data = load_iris()
X, y = data.data, data.target

# ---------------------------
# TRAIN MODEL
# ---------------------------
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

# ---------------------------
# STREAMLIT UI
# ---------------------------
st.title("Iris Flower Classifier")

st.write("Enter flower measurements below:")

sepal_length = st.number_input("Sepal Length", min_value=0.0)
sepal_width = st.number_input("Sepal Width", min_value=0.0)
petal_length = st.number_input("Petal Length", min_value=0.0)
petal_width = st.number_input("Petal Width", min_value=0.0)

# ---------------------------
# PREDICTION BUTTON
# ---------------------------
if st.button("Predict Flower Type"):

    user_data = [[sepal_length, sepal_width, petal_length, petal_width]]

    prediction = model.predict(user_data)

    predicted_class = data.target_names[prediction[0]]

    st.success(f"The flower is predicted to be: {predicted_class}")

# ---------------------------
# PCA VISUALIZATION
# ---------------------------
st.subheader("Dataset Visualization")

pca = PCA(n_components=2)
reduced_X = pca.fit_transform(X)

fig, ax = plt.subplots()

colors = ['r', 'b', 'g']
markers = ['x', 'D', '.']
labels = data.target_names

for i, (color, marker, label) in enumerate(zip(colors, markers, labels)):
    points = reduced_X[y == i]
    ax.scatter(points[:, 0], points[:, 1], c=color, marker=marker, label=label)

ax.set_title("PCA of Iris Dataset")
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.legend()

st.pyplot(fig)