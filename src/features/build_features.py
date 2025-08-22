import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
from FrequencyAbstraction import FourierTransformation
from sklearn.cluster import KMeans


# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df = pd.read_pickle("../../data/interim/02_outlires_removed_chauvenet.pkl")
df.info()

predictor_columns = list(df.columns[:6])

# Setting up fig settings for plots
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20,5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2

# --------------------------------------------------------------
# Dealing with missing values (imputation)
# --------------------------------------------------------------

for col in predictor_columns:
  df[col] = df[col].interpolate(method="linear")

df.info()

df[df["set"] == 45]["gyr_y"].plot(title=df[df["set"] == 45]["label"][0])

# --------------------------------------------------------------
# Calculating set duration
# --------------------------------------------------------------

for set in df["set"].unique():
  start = df[df["set"] == set].index[0]
  end = df[df["set"] == set].index[-1]
  duration = end - start
  df.loc[df["set"]==set, "duration"] = duration.seconds

duration_df = df.groupby(["category"])["duration"].mean()

# Since heavy category has 5 repetitions
float(duration_df.iloc[0] / 5) 

# Since medium category has 10 repetitions
float(duration_df.iloc[1] / 10)

# --------------------------------------------------------------
# Butterworth lowpass filter
# --------------------------------------------------------------

filter_df = df.copy()
filter_df.info()

lpfilter = LowPassFilter()

sampling_freq = 1000/200 # 200ms intervals
cutoff_freq = 1.2 # Adjust by inspecting plots

# Check low pass filter for 1 set
filter_df = lpfilter.low_pass_filter(
    filter_df,
    'gyr_y',
    sampling_freq,
    cutoff_freq,
    order=5
)

sub_s = filter_df[filter_df["set"] == 45]
print(sub_s["label"].iloc[0])

fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20,10))
ax[0].plot(sub_s["gyr_y"].reset_index(drop=True), label="raw data")
ax[1].plot(sub_s["gyr_y_lowpass"].reset_index(drop=True), label="butterworth filtered data")
ax[0].legend(loc="upper center", bbox_to_anchor=(0.5,1.15), fancybox=True)
ax[1].legend(loc="upper center", bbox_to_anchor=(0.5,1.15), fancybox=True)


for col in predictor_columns:
  filter_df = lpfilter.low_pass_filter(
    filter_df,
    col,
    sampling_freq,
    cutoff_freq,
    order=5
  )
  filter_df[col] = filter_df[col+"_lowpass"]
  del filter_df[col+"_lowpass"]

# --------------------------------------------------------------
# Principal component analysis PCA
# --------------------------------------------------------------

PCA = PrincipalComponentAnalysis()
df_pca = filter_df.copy()

pca_values = PCA.determine_pc_explained_variance(df_pca, predictor_columns)

#Selecting the number of principal components using elbow method
plt.figure(figsize=(10, 10))
plt.plot(range(1, len(predictor_columns)+1), pca_values, marker='o')
plt.xlabel('Principal Component Number')
plt.ylabel('Explained Variance')
plt.show()

df_pca = PCA.apply_pca(df_pca, predictor_columns, 3)

sub_set = df_pca[df_pca["set"] == 45]
sub_set[["pca_1", "pca_2", "pca_3"]].plot()


# --------------------------------------------------------------
# Sum of squares attributes
# --------------------------------------------------------------

df_sqaured = df_pca.copy()

acc_r = df_sqaured["acc_x"]**2 + df_sqaured["acc_y"]**2 + df_sqaured["acc_z"]**2
gyr_r = df_sqaured["gyr_x"]**2 + df_sqaured["gyr_y"]**2 + df_sqaured["gyr_z"]**2

df_sqaured["acc_r"] = np.sqrt(acc_r)
df_sqaured["gyr_r"] = np.sqrt(gyr_r)

subset = df_sqaured[df_sqaured["set"] == 45]
subset[["acc_r", "gyr_r"]].plot(subplots=True)

# --------------------------------------------------------------
# Temporal abstraction
# --------------------------------------------------------------

df_temporal = df_sqaured.copy()

numAbs = NumericalAbstraction()

predictor_columns = predictor_columns + ["acc_r", "gyr_r"]

# window size for compute rolling average
ws = int(1000/200) # window size of 1s

df_temporal = numAbs.abstract_numerical(
  df_temporal,
  predictor_columns,
  ws,
  "mean"
)

df_temporal = numAbs.abstract_numerical(
  df_temporal,
  predictor_columns,
  ws,
  "std"
)

# In the previous method 
# some values are aggregated using the values not belong to that set

df_temporal_list = []

for s in df_temporal["set"].unique():
  subset = df_temporal[df_temporal["set"]==s]
  subset = numAbs.abstract_numerical(
  subset,
  predictor_columns,
  ws,
  "mean"
  )

  subset = numAbs.abstract_numerical(
    subset,
    predictor_columns,
    ws,
    "std"
  )
  df_temporal_list.append(subset)

df_temporal = pd.concat(df_temporal_list)

df_temporal.info()

subset[["acc_y", "acc_y_temp_mean_ws_5", "acc_y_temp_std_ws_5"]].plot(subplots=True)
subset[["gyr_y", "gyr_y_temp_mean_ws_5", "gyr_y_temp_std_ws_5"]].plot()

# --------------------------------------------------------------
# Frequency features
# --------------------------------------------------------------

# Since FFT algorithms used in freq abs expect discrete functions as inputs
df_freq = df_temporal.copy().reset_index()
df_freq.info()
freqAbs = FourierTransformation()

fs = int(1000/200)
ws = int(2800/200)

#df_freq_dir = freqAbs.abstract_frequency(df_freq, predictor_columns, ws, fs)

df_freq_list = []

for s in df_freq["set"].unique():
  print(f"Apply DFT to set {s}")
  subset = df_freq[df_freq["set"] == s].reset_index(drop=True).copy()
  subset = freqAbs.abstract_frequency(subset,predictor_columns,ws,fs)
  df_freq_list.append(subset)

df_freq = pd.concat(df_freq_list).set_index("epoch (ms)", drop=True)

# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------

# Since we have created the new columns/features using a rolling window,
# The row values in the same column are highly correlated.
# We should somehow get rid of that to avoid overfitting of the models

df_freq = df_freq.dropna()

df_freq = df_freq.iloc[::2]

# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------

df_cluster = df_freq.copy()
cluster_columns = ["acc_x", "acc_y", "acc_z"]
k_values = range(2,10)
inertias = []

for k in k_values:
  subset = df_cluster[cluster_columns]
  kmeans = KMeans(n_clusters=k, n_init=20, random_state=0)
  cluster_label = kmeans.fit_predict(subset)
  inertias.append(kmeans.inertia_)


plt.figure(figsize=(10,10))
plt.plot(k_values, inertias, marker='o')
plt.xlabel("K vales")
plt.ylabel("Inertia")
plt.show()

# From elbow method number of clusers are 5
kmeans = KMeans(n_clusters=5, n_init=20, random_state=0)
subset = df_cluster[cluster_columns]
df_cluster["cluster"] = kmeans.fit_predict(subset)

# Plot the clusters in a 3d plot
fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(projection="3d")
for c in df_cluster["cluster"].unique():
  subset = df_cluster[df_cluster["cluster"]==c]
  ax.scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], label=c)
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
plt.legend()
plt.show()

# Plot the clusters in a 3d plot
fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(projection="3d")
for l in df_cluster["label"].unique():
  subset = df_cluster[df_cluster["label"]==l]
  ax.scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], label=l)
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
plt.legend()
plt.show()


# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

df_cluster.to_pickle("../../data/interim/03_data_features.pkl")