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

category_df = df.query("label=='squat'").query("participant=='A'").reset_index()

fig, ax = plt.subplots()
category_df.groupby(["category"])["acc_y"].plot()
ax.set_xlabel("samples")
ax.set_ylabel("acc_y")
ax.legend()

# --------------------------------------------------------------
# Compare participants
# --------------------------------------------------------------

participant_df = df[(df["label"]=="bench")].sort_values("participant").reset_index()
participant_df["participant"].unique()

fig, ax = plt.subplots()
participant_df.groupby(["participant"])["acc_y"].plot()
ax.set_xlabel("samples")
ax.set_ylabel("acc_y")
ax.legend()

# --------------------------------------------------------------
# Plot multiple axis
# --------------------------------------------------------------

label = "squat"
participant = "A"
all_axes_df = df.query(f"label == '{label}'").query(f"participant == '{participant}'").reset_index()

fig, ax = plt.subplots()
all_axes_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax)
ax.set_xlabel("samples")
ax.set_ylabel("acc")
ax.legend()

# --------------------------------------------------------------
# Create a loop to plot all combinations per sensor
# --------------------------------------------------------------

labels = df["label"].unique()
participants = df["participant"].unique()

for label in labels:
  for participant in participants:
    all_axes_df = (
    df.query(f"label == '{label}'")
    .query(f"participant == '{participant}'")
    .reset_index()
    )

    if len(all_axes_df)>0:
      fig, ax = plt.subplots()
      all_axes_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax)
      ax.set_xlabel("samples")
      ax.set_ylabel("acc")
      ax.set_title(f"{label} ({participant})".title())
      ax.legend()

for label in labels:
  for participant in participants:
    all_axes_df = (
    df.query(f"label == '{label}'")
    .query(f"participant == '{participant}'")
    .reset_index()
    )

    if len(all_axes_df)>0:
      fig, ax = plt.subplots()
      all_axes_df[["gyr_x", "gyr_y", "gyr_z"]].plot(ax=ax)
      ax.set_xlabel("samples")
      ax.set_ylabel("gyr")
      ax.set_title(f"{label} ({participant})".title())
      ax.legend()
    

# --------------------------------------------------------------
# Combine plots in one figure
# --------------------------------------------------------------

label = "row"
participant = "C"

combined_df = df.query(f"label == '{label}'").query(f"participant == '{participant}'").reset_index()

fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20,10))
combined_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax[0])
combined_df[["gyr_x", "gyr_y", "gyr_z"]].plot(ax=ax[1])

ax[0].legend(loc="upper center", bbox_to_anchor=(0.5,1.15), ncol=3, fancybox=True)
ax[1].legend(loc="upper center", bbox_to_anchor=(0.5,1.15), ncol=3, fancybox=True)
ax[1].set_xlabel("samples")

# --------------------------------------------------------------
# Loop over all combinations and export for both sensors
# --------------------------------------------------------------

labels = df["label"].unique()
participants = df["participant"].unique()

for label in labels:
  for participant in participants:
    combined_axes_plot = (
      df.query(f"label == '{label}'")
      .query(f" participant == '{participant}'")
      .reset_index()
    )

    if len(combined_axes_plot)>0:
      fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20,10))
      combined_axes_plot[["acc_x", "acc_y", "acc_z"]].plot(ax=ax[0])
      combined_axes_plot[["gyr_x", "gyr_y", "gyr_z"]].plot(ax=ax[1])

      ax[0].legend(loc="upper center", bbox_to_anchor=(0.5,1.15), ncol=3, fancybox=True)
      ax[1].legend(loc="upper center", bbox_to_anchor=(0.5,1.15), ncol=3, fancybox=True)
      ax[1].set_xlabel("samples")
      
      #plt.savefig(f"../../reports_and_figures/{label.title()} ({participant}).png")
      plt.show()