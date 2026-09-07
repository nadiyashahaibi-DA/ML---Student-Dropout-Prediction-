# ML---Student-Dropout-Prediction-
Machine Learning - Student Dropout Prediction
------------

Predicting Student Dropout & Academic Success

A machine learning project that predicts whether a student will Dropout, remain Enrolled, or Graduate — using only enrollment-time and first-semester data — so institutions can flag at-risk students early enough to intervene.

Team: Kseniia Iukhlina · Carla Fajula · Nadiya Al-Shahaibi Program: Data Analytics Bootcamp, Ironhack University

Problem Statement

Student attrition is costly for both students and institutions, but most warning signs only become visible after a student has already disengaged. This project asks: can we predict a student's eventual outcome using only the data available at enrollment and after their first semester — before mid-course intervention windows close?

Dataset
- Source: UCI Machine Learning Repository — "Predict Students' Dropout and Academic Success"
- Size: 4,424 students × 37 features (demographics, admission path, socio-economic status, and semester performance)
- Target distribution: Graduate 49.9% (2,209) · Dropout 32.1% (1,421) · Enrolled 17.9% (794) — moderately imbalanced, with Enrolled as the minority class
- Data quality: no missing values, no duplicate rows; IQR-based capping applied to outliers in age, admission grade, and semester performance (rows preserved rather than dropped)



Repository Structure
├── data.csv                                # Raw dataset (semicolon-delimited)
├── ML_Student_Dropout_nadiya.ipynb         # Data cleaning, EDA, chi-square/Cramér's V feature selection,
│                                            # KNN baseline, hyperparameter tuning, SMOTE experiment
├── 01_Load_clean_data_carla.ipynb          # Data loading and cleaning
├── 02_Random_forest_carla.ipynb            # Random Forest model, tuning, class-balancing experiments
├── Step_1_data_exploration_Kseniia.ipynb   # Exploratory data analysis
├── Step_2_model_selection_Kseniia.ipynb    # Gradient Boosting model, GridSearch tuning, SMOTE experiment
├── Presentation_Student_Dropout_Prediction.pptx  # Final stakeholder presentation
└── figures/                                # Exported chart images (KNN, Random Forest, Gradient Boosting)


Methodology
1. Clean & explore — validate categorical codes, check integrity, cap numeric outliers (IQR method)
2. Feature selection — chi-square tests (categorical features) and Cramér's V ranking to identify the strongest predictors of Target
3. Preprocessing — one-hot encode categorical features, standardize numeric features via ColumnTransformer
4. Modeling — three classifiers trained on the same 11 features (Course, Application mode/order, Daytime/evening attendance, Tuition fees up to date, Age at enrollment, Admission grade, and four 1st-semester curricular-unit metrics), stratified 80/20 train-test split
5. Tuning — GridSearchCV/RandomizedSearchCV (KNN, Random Forest) and manual + grid search (Gradient Boosting)
6. Imbalance handling — SMOTE and random oversampling tested against each model's unbalanced baseline
Results

Because classes are imbalanced, balanced accuracy and macro F1 are reported alongside plain accuracy.

Model	Accuracy	Balanced Accuracy	Macro F1	Notes
K-Nearest Neighbors	69%	60%	0.61	Baseline (k=5); tuning (k=13) improved CV score but not test performance
Random Forest	73.45%	62.05%	—	82.4% train accuracy → 9pp train-test gap after tuning
Gradient Boosting	75%	71%	0.67	Winning model — max_depth=3, n_estimators=100, learning_rate=0.10

- Gradient Boosting was selected as the best-performing model, leading on both balanced accuracy and macro F1 — the metrics that matter most on this imbalanced target.

- All three models share the same pattern: strong performance on Dropout and Graduate, but consistently weaker on Enrolled (F1 0.34–0.46), since early-semester signals for currently-enrolled students overlap with both eventual outcomes.

- On class balancing: SMOTE was tested against each model's baseline. It generally traded overall accuracy for better minority-class (Enrolled) recall — e.g., for Gradient Boosting, SMOTE lowered accuracy from 75% to 71% but improved Enrolled's recall from 33% to 57%. Given the project's goal of catching at-risk students early, this is a meaningful trade-off worth considering in production, even though the unbalanced models are reported as the headline results here.

- Strongest predictors (Random Forest feature importance): number of approved 1st-semester curricular units (29%), 1st-semester grade (17%), and tuition-fee status (10%) — confirming that early academic performance is the most useful signal available at this stage.

How to Run
bash
git clone https://github.com/nadiyashahaibi-DA/ML---Student-Dropout-Prediction-.git
cd ML---Student-Dropout-Prediction-
pip install pandas numpy scikit-learn matplotlib seaborn scipy imbalanced-learn jupyter
jupyter notebook

Run the notebooks in order: 
- 01_Load_clean_data_carla.ipynb → Step_1_data_exploration_Kseniia.ipynb → ML_Student_Dropout_nadiya.ipynb (KNN) → 02_Random_forest_carla.ipynb (Random Forest) → Step_2_model_selection_Kseniia.ipynb (Gradient Boosting).

Tech Stack

Python · pandas · NumPy · scikit-learn · imbalanced-learn (SMOTE) · matplotlib · seaborn · SciPy · Jupyter

Limitations & Next Steps

- Predictions are correlational, not causal — they should support human judgement, not replace it
- The Enrolled class remains hard to predict and merits further feature engineering or a dedicated model
- Next steps: additional student information, further model optimization, and an advisor-facing dashboard for real-world deployment

Team & Contributions

Member	Focus
- Nadiya Al-Shahaibi | 	Data cleaning, EDA, feature selection, KNN model, hyperparameter tuning, SMOTE
- Carla Fajula |	Data loading, Random Forest model, tuning, class-balancing experiments
- Kseniia Iukhlina |	EDA, Gradient Boosting model, hyperparameter tuning, SMOTE


License

No license file