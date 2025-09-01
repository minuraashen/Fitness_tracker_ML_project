# 🏋️ Fitness Tracker – Machine Learning with IMU Data

This project focuses on **automatic tracking of strength training exercises** using data from a wristband’s **accelerometer and gyroscope**. The ultimate goal is to develop models that can assist, like a personal trainer, in **tracking exercises** and **counting repetitions**.

---

<img width="1200" height="300" alt="image" src="https://github.com/user-attachments/assets/d0accff1-de31-4c38-9f26-b80b0bef7664" />

<p align="center">
    <em>Figure: Strength excersices to be classified</em>
</p>

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
- **Objective**: Develop supervised learning models for exercise classification and rep counting.
  
### 🎯 Objective 
- Explore, build, and evaluate models that:  
  - Track exercises  
  - Count repetitions  
### 🧠 Approach
- Supervised learning classification with different machine learning algorithms, evaluated by comparing accuracies  

---

## ⚙️ Methodology

### ✅ Data Preprocessing

  - Raw data was stored as separate CSV files for accelerometer and gyroscope sensors.  
  - All CSVs were concatenated into a single **pandas DataFrame**.  
  - Accelerometer and gyroscope streams were **merged** into a unified dataset based on timestamps.  

<p align="center">
   <img width="800" height="300" alt="image" src="https://github.com/minuraashen/Fitness_tracker_ML_project/blob/main/reports_and_figures/dataframe.png"/>
</p>
<p align="center">
    <em>Figure: Part of the preprocessed dataframe </em>
</p>

### ✅ Data Visualization

- **Time-Series Plots**: Visualized motion signals for each exercise.
- **Pattern Analysis**: Compared accelerometer and gyroscope data to uncover patterns.

<p align="center">
   <img width="800" height="300" alt="image" src="https://github.com/minuraashen/Fitness_tracker_ML_project/blob/main/reports_and_figures/Bench%20(A).png"/>
</p>
<p align="center">
    <em>Figure: Accelerometer data in a set (above) Gyroscope data in a set(below)</em>
</p>


### ✅ Outlier Detection

- Detected outliers using:
  - **IQR (Interquartile Range)**
  - **Chauvenet Method**
  - **Local Outlier Factor (LOF)**
- **Visualizations**

  - Outliers detected in accelerometer and gyroscope data using the **IQR method**:  

    <p align="center">
      <img src="https://github.com/minuraashen/Fitness_tracker_ML_project/blob/main/reports_and_figures/outliers_iqr_acc.png" width="400" height="600"/>
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
        <img width="400" alt="lof_acc" src="https://github.com/user-attachments/assets/3ea8c6e8-071f-4345-be03-57ddc7537d7f" />
        <img width="400"  alt="lof_gyr" src="https://github.com/user-attachments/assets/b1b396c2-0cc5-4d8c-959f-b104f890694e" />
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
   - Applied **Principal Component Analysis (PCA)** to reduce dimensionality of accelerometer + gyroscope feature space.  
   - Used the **Elbow Method** on explained variance ratio to determine the optimal number of principal components.  
   - This step helps remove redundant features while retaining maximum information for model training. 
     
     <p align="center">
        <img width="300" alt="pca using elbow" src="https://github.com/user-attachments/assets/0fa7eb87-8655-4d81-a27a-3b6435ef7838" />
        <img width="600" alt="plot_pc_for ohp_heavy" src="https://github.com/user-attachments/assets/979bea6c-b32e-48e7-b730-5d1bc889fd03" />
     </p>
     <p align="center">
        <em>Figure: Elbow method(left) and PCA example for one set of Overhead Press(right)</em>
     </p>

   - **Temporal Analysis**
     - Applied **rolling averages** to accelerometer and gyroscope signals.  
     - This smoothing technique reduces short-term noise and highlights temporal trends in the data.  
   - **Frequency analysis** using **Discrte Fourier Transform**
     - Applied **Discrete Fourier Transform (DFT)** on accelerometer and gyroscope signals.
     - By identifying dominant frequency components, we can distinguish between different exercise types and detect abnormal motion patterns.  
- **KMeans Clustering**
  - Optimal number of clusters was determined using the **Elbow Method**
  - Implemented clustering on the accelerometer data
  - Plotted **3D scatter plots** of clustered data to analyze separation between exercise movements

  <p align="center">
     <img width="400" alt="vizualize_clusters" src="https://github.com/user-attachments/assets/fa296fe9-5636-4ced-916b-8141a650f952" />
     <img width="400" alt="each_label_as_clusters" src="https://github.com/user-attachments/assets/2da8d168-916f-45dc-81f6-6dd8efe2f7b6" />
  </p>

  <p align="center">
    <em>Figure: 3D visualization of clustering (left) and cluster separation by labels (right)</em>
  </p>

     
### ✅ Predictive Modeling

- **Train-Test Split**: Prepared data for model training.
  - The preprocessed(feature added) dataset was divided into **training** and **testing** subsets to evaluate model generalization.  
  - **75/25 split** was used (75% training, 25% testing).  
  - Data shuffling was applied suing stratify method prior to splitting to avoid any sequence bias, since exercise data may be recorded in continuous sessions.  
  - This ensures that models learn robust patterns rather than memorizing specific exercise sequences.  
  <p align="center">
    <img width="500" alt="train_test_split" src="https://github.com/user-attachments/assets/482fc102-41a3-45c9-bbfb-cc434c28b091" />
  </p>
  <p align="center"><em>Bar plot representation of train test split</em></p>

- **Feature Selection**
  - Make feature subsets with basic features and also features we added in the feature engineering part.
  - Use **Forward Selection** algorithm to select best features that suit for train the dataset  
      1. Basic features  
      2. Basic features + sqaured sum of basic features
      3. Basic features + sqaured sum of basic features + temporal features
      4. All features
      5. Selected 10 features by forward selection  
    <p align="center">
     <img width="400" alt="Accuracy scores of features" src="https://github.com/user-attachments/assets/f56c0304-aa29-4856-89ef-207a1be55e11" />
   </p> 
   <P align="center"><em>Accuracy scores of selected featues vs number of features</em></P>
  - Use several classification algorithms with grid search in order to select the best feature subset and visualize the results for evaluation.

- **Model Training with Classification Algorithms**
  - Multiple **machine learning algorithms** were trained on different **feature subsets** derived from accelerometer and gyroscope signals.  
  - The following models were evaluated:
    - **Neural Network (NN)**  
    - **Random Forest (RF)**  
    - **Decision Tree (DT)**  
    - **k-Nearest Neighbors (kNN)**  
    - **Naïve Bayes (NB)**
      <p align="center">
      <img width="400" alt="accuracy_scores_all" src="https://github.com/user-attachments/assets/4e1e9153-7f73-4047-b729-0af4b15bd25c" />
      </p> 
      <p align="center"><em>Accuracy computed for each model–feature subset combination</em></p>

  - #### From the results the random forest and the neural network shows the almost same accuracy score for each feature subset. I have chosen Random Forest because it is a simpler model than a neural network.

- **Model training using selected algirithm and feature set and evaluation**
  - Random forest is used to train the model using feature set 4, that have all the features including tempotal features, frequency features and principal components.
  - I have achived a accuracy score of 99.38% for test data. A fantastic score.
  - To further evaluate model performance beyond accuracy, **confusion matrices** were generated.  
  - These visualizations highlight how well each model correctly classified exercises and where misclassifications occurred.
    <p align="center">
      <img width="400" alt="confusion_mat" src="https://github.com/user-attachments/assets/2ba141ba-0f5b-4faf-a3b8-a8216f1ce9e9" />
    </p>
    <p align="center"><em>Confusion matrix for model evaluation</em></p>

- **Train test split by participant**
  - In feature engineering we use rolling windows to calculate temporal features as well as frequency features. This may results data in neighbouring rows highly correlated. So the general train test split may not very fair for this situation. So I used to split data by participant, as training set (B, C, D, E) and test data set (participant A).
  
    <p align="center">
      <img width="600" alt="train_test_split_user" src="https://github.com/user-attachments/assets/e394c312-d480-444e-923b-a9286882ef7b" />
    </p>
    <p align="center"><em>Train test split by participant for fair seperation of data</em></p>

  - Evaluated using accuracy score and confusion matrix.
  - There also I have achieved accuracy score of 99.3% on test data.
    <p align="center>
      <img width="400"  alt="cm_participant" src="https://github.com/user-attachments/assets/64b93a5d-ba30-4cd7-8c0b-8805e3147cf4" />
    </p>
    <p align="center"><em>Confusion matrix for model evaluation</em></p>

- **Confusion matrices for Feedforward Nrural Network Model**
    <p align="center">
     <img width="400" alt="neu_net_all" src="https://github.com/user-attachments/assets/d4b81bef-a31e-4642-ac4e-d40f2d9e3940" />
      <img width="400" alt="neu_net_with_selected" src="https://github.com/user-attachments/assets/85b691d5-49ef-48db-befb-787f41f19a38" />

  </p>

  <p align="center">
    <em>Figure: CM for NN using all features (left) and Cm for NN using selected feature set(right)</em>
  </p>

### ✅ Count Repetitions
- To estimate the number of exercise repetitions, we use patterns in the sensor data after applying low-pass filtering:
  - Apply Low-Pass Filter → Removes high-frequency noise and keeps smooth movement signals.
  - Peak Detection → Each peak corresponds to one exercise repetition.
  - Evaluation → Compare predicted vs actual repetitions.
  <p align="center">
      <img width="600" alt="count_reps_medium" src="https://github.com/user-attachments/assets/0d085c53-d0df-44bb-a5ab-558e46f701df" />

    </p>
    <p align="center"><em>Count repetitions for a medium set</em></p>
    <p align="center">
      <img width="600" alt="count_reps_heavy" src="https://github.com/user-attachments/assets/e52223da-4808-4889-ab3e-6e9766810863" />
    </p>
    <p align="center"><em>Count repititions for a heavy set</em></p>
    <p align="center">
      <img width="600" alt="count_reps_evaluate" src="https://github.com/user-attachments/assets/60000423-9159-4dda-88fe-28599db80d6c" />
    </p>
    <p align="center"><em>Evaluation of count repetition prediction results with actual counts</em></p>


## 🛠️ Tech Stack

- **Python**  
- **NumPy / Pandas** – Data handling  
- **Matplotlib / Seaborn** – Visualization  
- **Scikit-learn** – Machine learning models  

---

## 📂 Repository Structure  
fitness-tracker-ml/    
├── data/                    
├── references/               
├── reports_and_figures/     
├── src/                      
│   ├── data/                 
│   ├── features/            
│   ├── models/             
│   ├── visualization/          
├── environment.yml          
├── requirements.txt          
└── README.md              


---

## 📌 Applications

- Automatic exercise classification (squats, bench press, deadlifts, etc.)  
- Real time **repitition counting**  
- Integration into **wearable fitness trackers**  

---

## References
- Hoogendoorn, M., & Funk, B. (2017). *Machine Learning for the Quantified Self: On the Art of Learning from Sensory Data* Springer(Chapter 7)
- ML4QS GitHub Repository: [https://github.com/davidstap/ML4QS](https://github.com/davidstap/ML4QS)

  

