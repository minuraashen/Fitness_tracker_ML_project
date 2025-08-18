<img width="1086" height="408" alt="image" src="https://github.com/user-attachments/assets/e7387614-12bc-4be3-ad3a-cc42ce092d69" /># 🏋️ Fitness Tracker – Machine Learning with IMU Data

This project focuses on **automatic tracking of strength training exercises** using data from a wristband’s **accelerometer and gyroscope**. The ultimate goal is to develop models that can assist, like a personal trainer, in **tracking exercises, counting repetitions, and detecting improper form**.

---

The aim of this project is to explore the potential of **context-aware fitness applications** within the strength training domain by analyzing wristband IMU data.  

- 📊 **Dataset:** Collected from **5 participants** performing various barbell exercises  
- 🎯 **Objective:** Explore, build, and evaluate models that:  
  - Track exercises  
  - Count repetitions  
  - Detect improper form  
- 🧠 **Approach:** Supervised learning classification with different machine learning algorithms, evaluated by comparing accuracies  

---

## ⚙️ Current Progress

✅ **Data Preprocessing**  
- Noise filtering & normalization  
- Segmentation into exercise repetitions  
Final dataframe structure
![](https://github.com/minuraashen/Fitness_tracker_ML_project/blob/main/reports_and_figures/dataframe.png)

✅ **Visualization**  
- Time-series plots for motion signals  
- Gyroscope vs Accelerometer pattern analysis
Accelerometer and gyroscope data for a perticular excersice and participant in a same plot
![](https://github.com/minuraashen/Fitness_tracker_ML_project/blob/main/reports_and_figures/Bench%20(A).png)

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

