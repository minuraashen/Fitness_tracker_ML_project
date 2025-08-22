import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction


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



# --------------------------------------------------------------
# Frequency features
# --------------------------------------------------------------


# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------


# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------


# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------