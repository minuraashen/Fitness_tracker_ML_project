import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from IPython.display import display

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
df = pd.read_pickle("../../data/interim/data_processed.pkl")

# --------------------------------------------------------------
# Plot single columns
# --------------------------------------------------------------

#Filter the only data belongs to set 1
set_df = df[df["set"] == 1]

#Using numerical indices
plt.plot(set_df["acc_y"].reset_index(drop=True))

# --------------------------------------------------------------
# Plot all exercises
# --------------------------------------------------------------

for label in df["label"].unique():
  sub_set = df[df["label"] == label]
  display(sub_set.head(5))
  plt.plot(sub_set["acc_x"].reset_index(drop=True), label=label)
  plt.legend()
  plt.show()

for label in df["label"].unique():
  sub_set = df[df["label"] == label]
  fig, ax = plt.subplots()
  plt.plot(sub_set[:100]["acc_x"].reset_index(drop=True), label=label)
  plt.legend()
  plt.show()
# --------------------------------------------------------------
# Adjust plot settings
# --------------------------------------------------------------

mpl.style.use("seaborn-v0_8-deep")
mpl.rcParams["figure.figsize"] = (20,5)
mpl.rcParams["figure.dpi"] = 100


# --------------------------------------------------------------
# Compare medium vs. heavy sets
# --------------------------------------------------------------

#Make a subset of data including squat data
squat_data = df.query("label == 'squat'").query("category == 'heavy'").query("participant == 'A'")
#As same as
squat_data_2 = df[(df["label"] == "squat") & (df["category"] == "heavy") & (df["participant"] == "A")]

# --------------------------------------------------------------
# Compare participants
# --------------------------------------------------------------


# --------------------------------------------------------------
# Plot multiple axis
# --------------------------------------------------------------


# --------------------------------------------------------------
# Create a loop to plot all combinations per sensor
# --------------------------------------------------------------


# --------------------------------------------------------------
# Combine plots in one figure
# --------------------------------------------------------------


# --------------------------------------------------------------
# Loop over all combinations and export for both sensors
# --------------------------------------------------------------