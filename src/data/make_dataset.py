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


acc_dataframe, gyr_dataframe = read_data_from_files(files, datapath)

# --------------------------------------------------------------
# Merging datasets
# --------------------------------------------------------------


# --------------------------------------------------------------
# Resample data (frequency conversion)
# --------------------------------------------------------------

# Accelerometer:    12.500HZ
# Gyroscope:        25.000Hz


# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------
