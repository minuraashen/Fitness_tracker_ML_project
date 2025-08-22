# 🏋️ Fitness Tracker – Machine Learning with IMU Data

This project focuses on **automatic tracking of strength training exercises** using data from a wristband’s **accelerometer and gyroscope**. The ultimate goal is to develop models that can assist, like a personal trainer, in **tracking exercises** and **counting repetitions**.

---

<img width="1200" height="300" alt="image" src="https://github.com/user-attachments/assets/d0accff1-de31-4c38-9f26-b80b0bef7664" />

---

## 🎯 Project Overview  

The aim of this project is to explore the potential of **context-aware fitness applications** within the strength training domain by analyzing wristband IMU data.  

### 📊 Dataset

- **Source**: IMU data from **5 participants** performing **5 barbell exercises**:
  - Squat
  - Deadlift
  - Overhead Press
  - Bench Press
  - Row
- **Data Type**: Accelerometer (x, y, z) and Gyroscope (x, y, z) readings
- **Objective**: Develop supervised learning models for exercise classification and rep counting
  
### 🎯 Objective 
- Explore, build, and evaluate models that:  
  - Track exercises  
  - Count repetitions  
### 🧠 Approach
- Supervised learning classification with different machine learning algorithms, evaluated by comparing accuracies  

---

## ⚙️ Current Progress

### ✅ Data Preprocessing

- **Noise Filtering & Normalization**: Cleaned raw IMU data to ensure consistency.
- **Segmentation**: Divided data into individual exercise repetitions.
- **Output**: Structured dataframe for further analysis.

<p align="center">
   <img width="800" height="300" alt="image" src="https://github.com/minuraashen/Fitness_tracker_ML_project/blob/main/reports_and_figures/dataframe.png"/>
</p>

### ✅ Data Visualization

- **Time-Series Plots**: Visualized motion signals for each exercise.
- **Pattern Analysis**: Compared accelerometer and gyroscope data to uncover patterns.

<p align="center">
   <img width="800" height="300" alt="image" src="https://github.com/minuraashen/Fitness_tracker_ML_project/blob/main/reports_and_figures/Bench%20(A).png"/>
</p>


### ✅ Outlier Detection

- Detected outliers using:
  - **IQR (Interquartile Range)**
  - **Chauvenet Method**
  - **Local Outlier Factor (LOF)**
- **Visualizations**

  - Outliers detected in accelerometer and gyroscope data using the **IQR method**:  

    <p align="center">
      <img src="https://github.com/minuraashen/Fitness_tracker_ML_project/blob/main/reports_and_figures/outliers_iqr_acc.png" width="400" />
      <img src="https://github.com/user-attachments/assets/7771a225-519b-4e55-93f5-29e5a9bd9b14" width="400" />
    </p>

    <p align="center"><em>Figure: Outliers in accelerometer (x-axis, left) and gyroscope (y-axis, right) using IQR</em></p>

  - Before applying the **Chauvenet method**, data normality was checked:  

    <p align="center">
      <img src="https://github.com/user-attachments/assets/d51fbd39-305a-4eb2-b809-c7e26ad699f4" width="400" />
      <img src="https://github.com/user-attachments/assets/57a5b71f-55a9-4114-965b-b9770fcf00bd" width="400" />
    </p>

    <p align="center"><em>Figure: Normality check for accelerometer (left) and gyroscope (right)</em></p>

  - Outliers detected using the **Chauvenet method**:  

    <p align="center">
      <img src="https://github.com/user-attachments/assets/f70308f1-b7b6-4d4a-abed-86cdb3525761" width="400" />
      <img src="https://github.com/user-attachments/assets/3d74893c-ea10-4262-a127-859be52d6f5d" width="400" />
    </p>

    <p align="center"><em>Figure: Outliers in accelerometer (left) and gyroscope (right) using Chauvenet</em></p>

- Local Outlier factor
     <p align="center">
        <img width="800" height="477" alt="lof_acc" src="https://github.com/user-attachments/assets/3ea8c6e8-071f-4345-be03-57ddc7537d7f" />
     </p>


### ✅ **Feature Engineering**  
- Remove subtle noise using butterworth filter. Adjust sampling frequency to have a better and smoother representation.
   - Better moothing
     
  <p align="center">
    <img width="400" alt="better_filtering" src="https://github.com/user-attachments/assets/3dd09dc2-e175-422c-8a84-29ef9f60226d" />
    <img width="400" alt="too_much_filtered" src="https://github.com/user-attachments/assets/2cca8ce7-a4eb-42b7-b15c-40698862da0e" />
  </p>

  <p align="center">
    <em>Figure: Balanced filtering (left) vs. excessive smoothing (right)</em>
  </p>

- Add Features to the dataset
   - **Principal Component Analysis** for dimentionality reduction. Determine the number of principal components required using **Elbow method**
     
     <p align="center">
        <img width="300" alt="pca using elbow" src="https://github.com/user-attachments/assets/0fa7eb87-8655-4d81-a27a-3b6435ef7838" />
        <img width="600", height=250mm alt="plot_pc_for ohp_heavy" src="https://github.com/user-attachments/assets/979bea6c-b32e-48e7-b730-5d1bc889fd03" />
     </p>
     <p align="center">
        <em>Figure: Elbow method(left) and PCA example for one set of Overhead Press(right)</em>
     </p>

   - **Temporal Analysis**
   - **Frequency analysis** using **Discrte Fourier Transform**
- **Clustering**
  - Visualization of clusters  

  <p align="center">
     <img width="400" alt="vizualize_clusters" src="https://github.com/user-attachments/assets/fa296fe9-5636-4ced-916b-8141a650f952" />
     <img width="400" alt="each_label_as_clusters" src="https://github.com/user-attachments/assets/2da8d168-916f-45dc-81f6-6dd8efe2f7b6" />
  </p>

  <p align="center">
    <em>Figure: 3D visualization of clustering (left) and cluster separation by labels (right)</em>
  </p>

     
### ✅ Predictive Modeling

- **Train-Test Split**: Prepared data for model training.
  <p align="center">
    <img src="https://github.com/user-attachments/assets/1231086f-f47a-4e90-b368-a23dc2169a73" alt="Train-Test Split" width="600"/>
  </p>

🚧 **Next Steps**   
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

