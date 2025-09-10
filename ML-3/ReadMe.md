Email Spam or Ham Classification

Objective

The objective of this project is to classify emails as **Spam or Ham (Not Spam)** using three machine learning algorithms: **Naive Bayes, K-Nearest Neighbors (KNN), and Support Vector Machine (SVM)**. The models are evaluated using **accuracy metrics** and **K-Fold cross-validation**, with additional visualizations such as **Confusion Matrices** and **ROC Curves**.

This experiment demonstrates how different classification algorithms can be applied and compared for a real-world problem in text-based email filtering.


Files Included

* **naive\_bayes\_bernoulli.py** – Implements Bernoulli Naive Bayes with evaluation metrics and ROC curve plotting.
* **naive\_bayes\_multinomial.py** – Implements Multinomial Naive Bayes for discrete features.
* **naive\_bayes\_gaussian.py** – Implements Gaussian Naive Bayes for continuous features.
* **knn\_classifier.py** – Implements K-Nearest Neighbors with varying values of *k*, KDTree vs BallTree comparison, and GridSearch for best parameters.
* **svm\_classifier.py** – Implements Support Vector Machine with different kernels (**Linear, Polynomial, RBF, Sigmoid**) and evaluates their performance.
* **spambase\_csv.csv** – Dataset used for training and testing the classifiers (ensure correct path).



Naive Bayes (Bernoulli, Multinomial, Gaussian)

* Loads and preprocesses the dataset (handles missing values, encodes categorical variables).
* Trains Naive Bayes classifiers on training data.
* Evaluates using Accuracy, Precision, Recall, F1 Score.
* Performs 5-Fold Cross-Validation.
* Plots Confusion Matrix and ROC Curve.

K-Nearest Neighbors (KNN)

* Implements KNN with multiple values of *k* (1, 3, 5, 7, 9, 11).
* Compares performance across **auto, ball\_tree, and kd\_tree** algorithms.
* Uses **GridSearchCV** to find the best hyperparameters.
* Evaluates with metrics and confusion matrices.

Support Vector Machine (SVM)

* Trains models with **Linear, Polynomial, RBF, and Sigmoid kernels**.
* Evaluates on validation and test sets with Accuracy, Precision, Recall, F1 Score.
* Performs 5-Fold Cross-Validation.
* Plots ROC Curves for each kernel.

 How to Run

1. Make sure you have Python **3.x** installed.
2. Install required libraries:

   ```bash
   pip install pandas numpy seaborn matplotlib scikit-learn
   ```
3. Place the dataset (`spambase_csv.csv`) in the correct path as referenced in the scripts.
4. Run any script from the terminal:

   ```bash
   python naive_bayes_bernoulli.py
   python knn_classifier.py
   python svm_classifier.py
   ```

---
Requirements

* Python 3.x
* numpy
* pandas
* seaborn
* matplotlib
* scikit-learn

---

Notes

* Ensure the dataset file path is updated in each script before running.
* The scripts include **Confusion Matrix plots** and **ROC Curve plots** for better visualization of classifier performance.
* For KNN and SVM, hyperparameter tuning is demonstrated to optimize performance.

---

