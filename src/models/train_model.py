import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from LearningAlgorithms import ClassificationAlgorithms
import seaborn as sns
import itertools
from sklearn.metrics import accuracy_score, confusion_matrix


# Plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2

df = pd.read_pickle("../../data/interim/03_data_features.pkl")

# --------------------------------------------------------------
# Create a training and test set
# --------------------------------------------------------------

df_train = df.drop(["participant", "category", "set", "duration"] , axis=1)

X = df_train.drop("label", axis=1)
Y = df_train["label"]

x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size = 0.25, random_state=42, stratify=Y
)

fig, ax = plt.subplots(figsize=(10,5))
labels = sorted(df["label"].unique())

df_train["label"].value_counts().reindex(labels).plot(
    kind="bar", ax=ax, color="lightblue", label="Total"
)
y_train.value_counts().reindex(labels).plot(
    kind="bar", ax=ax, color="dodgerblue", label="Train"
)
y_test.value_counts().reindex(labels).plot(
    kind="bar", ax=ax, color="royalblue", label="Test"
)
plt.legend()
plt.show()


# --------------------------------------------------------------
# Split feature subsets
# --------------------------------------------------------------

basic_features = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]
square_features = ["acc_r", "gyr_r"]
pca_features = ["pca_1", "pca_2", "pca_3"]
time_features = [f for f in df_train.columns if "_temp_" in f]
freq_features = [f for f in df_train.columns if ("_freq" in f) or ("pse" in f)]
cluster_features = ["cluster"]

print("basic Features:", len(basic_features))
print("Square Features:", len(square_features))
print("PCA Features:", len(pca_features))
print("Time Features:", len(time_features))
print("Frequency Features:", len(freq_features))
print("Cluster Features:", len(cluster_features))

feature_set_1 = list(set(basic_features))
feature_set_2 = list(set(basic_features + square_features + pca_features))
feature_set_3 = list(set(feature_set_2 + time_features))
feature_set_4 = list(set(feature_set_3 + freq_features + cluster_features))


# --------------------------------------------------------------
# Perform forward feature selection using simple decision tree
# --------------------------------------------------------------

learner = ClassificationAlgorithms()

max_features = 10

selected_features, ordered_features, ordered_scores = learner.forward_selection(
    max_features,
    x_train,
    y_train
)

selected_features = ['acc_z_freq_0.0_Hz_ws_14',
 'acc_x_freq_0.0_Hz_ws_14',
 'gyr_r_freq_0.0_Hz_ws_14',
 'acc_z',
 'acc_y_freq_0.0_Hz_ws_14',
 'gyr_r_freq_1.429_Hz_ws_14',
 'gyr_r_max_freq',
 'gyr_x_max_freq',
 'acc_z_freq_2.143_Hz_ws_14',
 'acc_x_max_freq']

plt.figure(figsize=(10,8))
plt.plot(range(1, len(ordered_scores)+1), ordered_scores, marker='o')
plt.xlabel("Number of Features")
plt.ylabel("Accuracy Score")
plt.show()

# --------------------------------------------------------------
# Grid search for best hyperparameters and model selection
# --------------------------------------------------------------

feature_sets = [
  feature_set_1,
  feature_set_2,
  feature_set_3,
  feature_set_4,
  selected_features
]

feature_names = [
  "Feature Set 1",
  "feature Set 2",
  "Feature Set 3",
  "Feature Set 4",
  "Selected Features"
]

# Grid search for chooisng best hyperparameters
iterations = 1
score_df = pd.DataFrame()

# Loop through each model and feature set

for i, f in enumerate(feature_names):
  print("Feature Set:", i)
  selected_train_X = x_train[feature_sets[i]]
  selected_test_X = x_test[feature_sets[i]]

  # First run non deterministic classifiers to average their scores
  performance_test_nn = 0
  performance_test_rf = 0

  for it in range(0, iterations):
    print("\tTraining nueral network", it)
    (
      class_train_y,
      class_test_y,
      class_train_prob_y,
      class_test_prob_y
    ) = learner.feedforward_neural_network(
        selected_train_X,
        y_train, 
        selected_test_X,
        gridsearch=False
    )
    performance_test_nn += accuracy_score(y_test, class_test_y)

    print("\tTraining random forest", it)
    (
      class_train_y,
      class_test_y,
      class_train_prob_y,
      class_test_prob_y
    ) = learner.random_forest(
      selected_train_X,
      y_train,
      selected_test_X,
      gridsearch=True
    )
    performance_test_rf += accuracy_score(y_test, class_test_y)

    performance_test_nn = performance_test_nn/iterations
    performance_test_rf = performance_test_rf/iterations

    # And we run our deterministic classifiers:
    print("\tTraining KNN")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.k_nearest_neighbor(
        selected_train_X, y_train, selected_test_X, gridsearch=True
    )
    performance_test_knn = accuracy_score(y_test, class_test_y)

    print("\tTraining decision tree")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.decision_tree(
        selected_train_X, y_train, selected_test_X, gridsearch=True
    )
    performance_test_dt = accuracy_score(y_test, class_test_y)

    print("\tTraining naive bayes")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.naive_bayes(selected_train_X, y_train, selected_test_X)

    performance_test_nb = accuracy_score(y_test, class_test_y)

    # Save the results in the dataframe
    models = ["NN", "RF", "KNN", "DT", "NB"]
    new_scores = pd.DataFrame(
      {
        "model": models,
        "feature_set": f,
        "accuracy": [
          performance_test_nn,
          performance_test_rf,
          performance_test_knn,
          performance_test_dt,
          performance_test_nb,
        ]
      }
    )

    score_df = pd.concat([score_df, new_scores])


# --------------------------------------------------------------
# Create a grouped bar plot to compare the results
# --------------------------------------------------------------

score_df.sort_values(by="accuracy", ascending=False)

plt.figure(figsize=(10,10))
sns.barplot(x="model", y="accuracy", hue="feature_set", data=score_df)
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.ylim(0.7,1)
plt.legend(loc="lower right")
plt.show()

# --------------------------------------------------------------
# Select best model and evaluate results
# --------------------------------------------------------------

(
  class_train_y, 
  class_test_y,
  class_train_prob_y, 
  class_test_prob_y
) = learner.random_forest(
  x_train[feature_set_4],
  y_train,
  x_test[feature_set_4],
  gridsearch=True
)

accuracy = accuracy_score(y_test, class_test_y)

classes = class_test_prob_y.columns

cm = confusion_matrix(y_true=y_test, y_pred=class_test_y, labels=classes)

plt.figure(figsize=(10,10))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",xticklabels=classes, yticklabels=classes)
plt.title("Confusion Matrix")

# --------------------------------------------------------------
# Select train and test data based on participant
# --------------------------------------------------------------

# Since in temporal analysis we use rolling average, 
# There may be significat overlap between each rows.
# So if we perform train_test_split it may not be generalize the model to unseen data
# So we split data by users

participant_df = df.drop(["category", "set","duration"], axis=1)

X_train = participant_df[participant_df["participant"] != "A"].drop(["label"], axis=1)
Y_train = participant_df[participant_df["participant"] != "A"]["label"]

X_test = participant_df[participant_df["participant"] == "A"].drop(["label"], axis=1)
Y_test = participant_df[participant_df["participant"] == "A"]["label"]

X_train = X_train.drop(["participant"], axis=1)
X_test = X_test.drop(["participant"], axis=1)

fig, ax = plt.subplots(figsize=(10,5))
labels = sorted(df["label"].unique())
df_train["label"].value_counts().reindex(labels).plot(kind="bar", ax=ax, color="lightblue", label="Total")
Y_train.value_counts().reindex(labels).plot(kind="bar", ax=ax, color="dodgerblue", label="Train")
Y_test.value_counts().reindex(labels).plot(kind="bar", ax=ax, color="royalblue", label="Test")
plt.legend()
plt.show()

# --------------------------------------------------------------
# Use best model again and evaluate results
# --------------------------------------------------------------

(
  class_train_y, 
  class_test_y, 
  class_train_prob_y, 
  class_test_prob_y
) = learner.random_forest(
  X_train[feature_set_4], Y_train, X_test[feature_set_4], gridsearch=True
)
 
accuracy = accuracy_score(Y_test, class_test_y)

classes = class_test_prob_y.columns

cm = confusion_matrix(y_true=Y_test, y_pred=class_test_y, labels=classes)

plt.figure(figsize=(11,10))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=classes, yticklabels=classes)
plt.title("Confusion Matrix")


# --------------------------------------------------------------
# Try nueral network model with the selected features
# --------------------------------------------------------------


(
  class_train_y, 
  class_test_y, 
  class_train_prob_y, 
  class_test_prob_y
) = learner.feedforward_neural_network(
  X_train, Y_train, X_test, gridsearch=False
)
 
accuracy = accuracy_score(Y_test, class_test_y)

classes = class_test_prob_y.columns

cm = confusion_matrix(y_true=Y_test, y_pred=class_test_y, labels=classes)

plt.figure(figsize=(10,10))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes)
plt.yticks(tick_marks, classes)

thresh = cm.max() / 2
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
  plt.text(
    i,
    j,
    format(cm[i,j]),
    horizontalalignment = "center",
    color="white" if cm[i,j] > thresh else "black"
  )

plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.grid(False)
