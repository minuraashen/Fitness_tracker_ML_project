#Import Essential Libraries
import pandas as pd
from glob import glob

def read_data_from_files(files, datapath):
	'''
	Fuction for reading Accelerometer and Gyroscope data in csv files
	and convert them into two data frames using pandas. 
	Time data in UNIX format is converted into datetime objets and 
	dataframes are indexed using that datetime in order to make a time series. 
	Unnecessary time data are removed.
	'''
	acc_dataframe = pd.DataFrame()
	gyr_dataframe = pd.DataFrame()

	acc_set = 1
	gyr_set = 1

	for f in files:
		participant = f.split("-")[0].replace(datapath, "")
		label = f.split("-")[1]
		category = f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")

		df =  pd.read_csv(f)

		df["participant"] = participant
		df["label"] = label
		df["category"] = category

		if "Accelerometer" in f: 
			df["set"] = acc_set
			acc_set += 1
			acc_dataframe = pd.concat([ acc_dataframe, df ])
			

		if "Gyroscope" in f: 
			df["set"] = gyr_set
			gyr_set += 1
			gyr_dataframe = pd.concat([ gyr_dataframe, df ])

	acc_dataframe.index = pd.to_datetime(acc_dataframe["epoch (ms)"], unit="ms")
	gyr_dataframe.index = pd.to_datetime(gyr_dataframe["epoch (ms)"], unit="ms")

	del acc_dataframe["epoch (ms)"]
	del acc_dataframe["time (01:00)"]
	del acc_dataframe["elapsed (s)"]

	del gyr_dataframe["epoch (ms)"]
	del gyr_dataframe["time (01:00)"]
	del gyr_dataframe["elapsed (s)"]

	return acc_dataframe, gyr_dataframe
		

#Declare initial variables of raw data and datapath 
files = glob("../../data/raw/MetaMotion/*.csv")
datapath = "../../data/raw/MetaMotion\\"

#Making the two dataframes using accelerometer and gyroscope data
acc_dataframe, gyr_dataframe = read_data_from_files(files, datapath)

#Removed replicated information and restructure the dataframe
data_frame_merged = pd.concat([acc_dataframe.iloc[:,:3], gyr_dataframe], axis=1)

#Rename Columns
data_frame_merged.columns=[
	"acc_x",
	"acc_y",
	"acc_z",
	"gyr_x",
	"gyr_y",
	"gyr_z",
	"participant",
	"label",
	"category",
	"set"
]

#Aggregation method for each column
sampling = {
	"acc_x": "mean",
	"acc_y": "mean",
	"acc_z": "mean",
	"gyr_x": "mean",
	"gyr_y": "mean",
	"gyr_z": "mean",
	"participant": "last",
	"label": "last",
	"category": "last",
	"set": "last"
}

#Split the data by days
days = [g for n,g in data_frame_merged.groupby(pd.Grouper(freq="D"))]

#Resampled data recorded in each day, samples per every 200ms
data_resampled = pd.concat([df.resample(rule="200ms").apply(sampling).dropna() for df in days])

data_resampled["set"] = data_resampled["set"].astype("int")


# Converted to the pickle format and export the processed data
# Here we use pickle format as we can import this data as same as we exported
data_resampled.to_pickle("../../data/interim/data_processed.pkl")
