# 🏋️ Fitness Tracker – Machine Learning with IMU Data

This project focuses on **automatic tracking of strength training exercises** using data from a wristband’s **accelerometer and gyroscope**. The ultimate goal is to develop models that can assist, like a personal trainer, in **tracking exercises** and **counting repetitions**.

---

<img width="1200" height="300" alt="image" src="https://github.com/user-attachments/assets/d0accff1-de31-4c38-9f26-b80b0bef7664" />

---

The aim of this project is to explore the potential of **context-aware fitness applications** within the strength training domain by analyzing wristband IMU data.  

- 📊 **Dataset:** Collected from **5 participants** performing basic 5 barbell exercises:
   - Squat
   - Deadlift
   - Overhead Press
   - Bench Press
   - Row
- 🎯 **Objective:** Explore, build, and evaluate models that:  
  - Track exercises  
  - Count repetitions  
- 🧠 **Approach:** Supervised learning classification with different machine learning algorithms, evaluated by comparing accuracies  

---

## ⚙️ Current Progress

✅ **Data Preprocessing**  
- Noise filtering & normalization  
- Segmentation into exercise repetitions  
Final dataframe structure

![](https://github.com/minuraashen/Fitness_tracker_ML_project/blob/main/reports_and_figures/dataframe.png)

✅ **Visualization of Data**  
- Time-series plots for motion signals  
- Gyroscope vs Accelerometer pattern analysis
Accelerometer and gyroscope data for a perticular excersice and participant in a same plot

![](https://github.com/minuraashen/Fitness_tracker_ML_project/blob/main/reports_and_figures/Bench%20(A).png)

✅ **Detecting Outliers**  
- Detecting outliers in the sensor data using three methods.
  - Using IQR(Inter Quartile Range)
  - Chauvenet Method
  - Local Outlier Factor
- Below Figures are illlustrated the outliers in both accelerometer and gyroscope data marked using IQR method.
   - Outliers in Accelerometer data in x direction
     
   <img width="1840" height="477" alt="outliers_iqr_acc" src="https://github.com/minuraashen/Fitness_tracker_ML_project/blob/main/reports_and_figures/outliers_iqr_acc.png"/>
   - Outliers in Gyroscope data in x direction
     
   <img width="1846" height="477" alt="outliers_iqr_gyr" src="https://github.com/user-attachments/assets/7771a225-519b-4e55-93f5-29e5a9bd9b14" />
   
- To apply the chauvenet method for detecting outliers we need to have normally distributed dataset. It seems somehow close to normal distributions.
  
   <img width="1842" height="598" alt="check_normal_for_chauvenet_acc" src="https://github.com/user-attachments/assets/d51fbd39-305a-4eb2-b809-c7e26ad699f4" />
   
   <img width="1842" height="598" alt="check_normal_for_chauvenet_gyr" src="https://github.com/user-attachments/assets/57a5b71f-55a9-4114-965b-b9770fcf00bd" />



✅ **Feature Engineering**  
- Remove subtle noise 

🚧 **Next Steps**  
- Feature extraction (time-domain & frequency-domain features)  
- Train ML models (Random Forest, CNN, LSTM)  
- Compare performance & generalization  

---

## 🛠️ Tech Stack

- **Python**  
- **NumPy / Pandas** – Data handling  
- **Matplotlib / Seaborn** – Visualization  
- **Scikit-learn / TensorFlow (planned)** – Machine learning models  

---

## 📂 Repository Structure  
fitness-tracker-ml/  
├── data/ # Raw & preprocessed datasets  
├── notebooks/ # Jupyter notebooks for experiments  
├── src/ # Source code (preprocessing, features, models)  
├── images/ # Plots & visualizations  
└── README.md  


---

## 📌 Applications

- Automatic exercise classification (squats, bench press, curls, etc.)  
- Real-time **rep counting**  
- Detection of **improper form** for injury prevention  
- Integration into **wearable fitness trackers**  

---

