import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from IPython.display import display

# Load data
df = pd.read_pickle("../../data/interim/data_processed.pkl")

# Loop over all combinations and export for both sensors
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