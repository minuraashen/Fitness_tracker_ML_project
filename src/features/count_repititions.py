import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter
from scipy.signal import argrelextrema
from sklearn.metrics import mean_absolute_error

pd.options.mode.chained_assignment = None


# Plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2


# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
 
df = pd.read_pickle("../../data/interim/data_processed.pkl")
df = df[df["label"] != "rest"]

df["label"].unique()

acc_r = df["acc_x"]**2 + df["acc_y"]**2 + df["acc_z"]**2
gyr_r = df["gyr_x"]**2 + df["gyr_y"]**2 + df["gyr_z"]**2

df["acc_r"] = np.sqrt(acc_r)
df["gyr_r"] = np.sqrt(gyr_r)

# --------------------------------------------------------------
# Split data
# --------------------------------------------------------------

bench_df = df[df["label"] == "bench"]
ohp_df = df[df["label"] == "ohp"]
squat_df = df[df["label"] == "squat"]
dead_df = df[df["label"] == "dead"]
row_df = df[df["label"] == "row"]

# --------------------------------------------------------------
# Visualize data to identify patterns
# --------------------------------------------------------------

plot_df = bench_df

plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["acc_x"].plot()
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["acc_y"].plot()
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["acc_z"].plot()
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["acc_r"].plot()

plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["gyr_x"].plot()
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["gyr_y"].plot()
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["gyr_z"].plot()
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["gyr_r"].plot()

# --------------------------------------------------------------
# Configure LowPassFilter
# --------------------------------------------------------------

fs = 1000/200

lowPass = LowPassFilter()

# --------------------------------------------------------------
# Apply and tweak LowPassFilter
# --------------------------------------------------------------

bench_set = bench_df[bench_df["set"] == bench_df["set"].unique()[0]]
ohp_set = ohp_df[ohp_df["set"] == ohp_df["set"].unique()[11]]
squat_set = squat_df[squat_df["set"] == squat_df["set"].unique()[6]]
dead_set = dead_df[dead_df["set"] == dead_df["set"].unique()[5]]
row_set = row_df[row_df["set"] == row_df["set"].unique()[-7]]



dead_set_lowpass = lowPass.low_pass_filter(ohp_set, "acc_r", fs, cutoff_frequency=0.4, order=5)
dead_set_lowpass["acc_r"].plot()
dead_set_lowpass["acc_r"+"_lowpass"].plot()

columns = df.columns.drop(["participant","label", "category", "set"])

for col in columns:
  fig, ax = plt.subplots(2,1, sharex=True)
  dead_set_lowpass = lowPass.low_pass_filter(dead_set, col , fs, cutoff_frequency=0.4, order=5)
  indices_of_peaks = argrelextrema(dead_set_lowpass[col+"_lowpass"].values, np.greater)
  peaks = dead_set_lowpass.iloc[indices_of_peaks]
  dead_set_lowpass[col].plot(ax=ax[0], color="green")
  dead_set_lowpass[col+"_lowpass"].plot(ax=ax[1])
  peaks[col+"_lowpass"].plot(ax=ax[1], marker='o', color="red", linestyle="none")
  ax[0].set_title("ohp "+col)
  plt.show()
  print(len(peaks))
  
counr_rep_dict = {
  "bench_m": ("acc_r", 0.4),
  "bench_h": ("gyr_x", 0.4),
  "ohp_m": ("acc_r", 0.5),
  "ohp_h": ("gyr_z", 0.4),
  "squat_m": ("acc_r", 0.35),
  "squat_h": ("acc_r", 0.35),
  "dead_m": ("gyr_r", 0.4),
  "dead_h": ("gyr_x", 0.4),
  "row_m": ("gyr_x", 0.4),
  "row_h": ("gyr_x", 0.7)
}



# --------------------------------------------------------------
# Create function to count repetitions
# --------------------------------------------------------------

def count_reps(dataset, col, fs, fc):
  fig, ax = plt.subplots(2,1, sharex=True)
  lowpass_set = lowPass.low_pass_filter(
    dataset, col, fs, fc, order=5
  )
  indices_of_peaks = argrelextrema(lowpass_set[col+"_lowpass"].values, np.greater)
  peaks = lowpass_set.iloc[indices_of_peaks]
  lowpass_set[col].plot(ax=ax[0], color="green")
  lowpass_set[col+"_lowpass"].plot(ax=ax[1])
  peaks[col+"_lowpass"].plot(ax=ax[1], marker='o', color="red", linestyle="none")
  ax[0].set_title(f"{lowpass_set["label"].iloc[0]} {col} {lowpass_set["set"].iloc[0]} {lowpass_set["category"].iloc[0]}")
  plt.show()
  del dataset[col+"_lowpass"]
  print(len(peaks))
  return len(peaks)

def select_category_count_reps(dataset, col_m, col_h, fc_m, fc_h):
  if bool(dataset["category"].unique() == "heavy"):
    return count_reps(dataset, col_h, fs, fc_h)
  else:
    return count_reps(dataset, col_m, fs, fc_m)

for s in df["set"].unique():
  dataset = df[df["set"] == s]

  if dataset["label"].iloc[0] == "bench":
    reps = select_category_count_reps(dataset, "acc_r", "gyr_x", 0.4, 0.4)
  elif dataset["label"].iloc[0] == "ohp":
    reps = select_category_count_reps(dataset, "acc_r", "gyr_z", 0.5, 0.4)
  elif dataset["label"].iloc[0] == "squat":
    reps = select_category_count_reps(dataset, "acc_r", "acc_r", 0.35, 0.35)
  elif dataset["label"].iloc[0] == "dead":
    reps = select_category_count_reps(dataset, "gyr_r", "gyr_x", 0.4, 0.4)
  else:
    reps = select_category_count_reps(dataset, "gyr_x", "gyr_x", 0.5, 0.7)

  df.loc[df["set"] == s, "reps_count"] = reps


# --------------------------------------------------------------
# Create benchmark dataframe
# --------------------------------------------------------------

df["actual_reps"] = df["category"].apply(lambda x:5 if x == "heavy" else 10)
rep_df = df.groupby(["label", "category", "set"])[["actual_reps", "reps_count"]].max().reset_index()

# --------------------------------------------------------------
# Evaluate the results
# --------------------------------------------------------------

error = round(mean_absolute_error(rep_df["actual_reps"], rep_df["reps_count"]),3)

# Create a barplot
d = rep_df.groupby(["label", "category"])[["actual_reps", "reps_count"]].mean().plot.bar()
