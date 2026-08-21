# 🏅 Athlete Winning Prediction Using Machine Learning

## 📖 Overview

This project uses **Machine Learning** to predict the number of medals an Olympic team is likely to win based on historical team and athlete data.

The project explores the relationship between factors such as the **number of athletes**, **previous medal performance**, and the number of medals won. A **Linear Regression** model is trained on historical data and then used to make predictions on unseen data.

The project also includes exploratory data analysis and visualizations to better understand the relationship between team characteristics and medal performance.

---

## 🎯 Project Objective

The main objective of this project is to demonstrate how machine learning can be applied to historical Olympic data to estimate future medal performance.

The model uses information available from previous Olympic competitions to learn patterns that may help predict the number of medals a team could win.

The project follows this general workflow:

```text
Historical Olympic Data
        ↓
Data Loading
        ↓
Data Exploration
        ↓
Data Cleaning
        ↓
Train/Test Split
        ↓
Linear Regression Model
        ↓
Medal Predictions
        ↓
Model Evaluation
        ↓
Error Analysis & Visualization
```

---

## 📂 Project Structure

```text
Athlete-winning-prediction-using-ML/
│
├── Figure_1.png
├── Figure_2.png
├── Figure_3.png
├── machine_learning.py
├── teams.csv
└── README.md
```

### Files

| File                  | Description                                                                                          |
| --------------------- | ---------------------------------------------------------------------------------------------------- |
| `machine_learning.py` | Main Python script containing the data analysis, model training, prediction, and evaluation process. |
| `teams.csv`           | Dataset containing historical Olympic team information and medal results.                            |
| `Figure_1.png`        | Visualization generated during the exploratory analysis.                                             |
| `Figure_2.png`        | Visualization showing relationships within the dataset.                                              |
| `Figure_3.png`        | Visualization of model/error analysis.                                                               |
| `README.md`           | Documentation for the project.                                                                       |

---

# 📊 Dataset

The `teams.csv` file contains historical Olympic team-level information.

Important columns include:

| Column          | Description                                           |
| --------------- | ----------------------------------------------------- |
| `team`          | Team/country code                                     |
| `country`       | Country name                                          |
| `year`          | Olympic year                                          |
| `events`        | Number of events participated in                      |
| `athletes`      | Number of athletes representing the team              |
| `age`           | Average athlete age                                   |
| `height`        | Average athlete height                                |
| `weight`        | Average athlete weight                                |
| `medals`        | Number of medals won                                  |
| `prev_medals`   | Number of medals won in the previous Olympic Games    |
| `prev_3_medals` | Average/measure of medals from previous Olympic Games |

The dataset contains historical observations across multiple Olympic years.

---

# 🔎 Exploratory Data Analysis

The project first loads the dataset using **Pandas** and selects the variables relevant to the analysis.

```python
teams = pd.read_csv("teams.csv")

teams = teams[
    [
        "team",
        "country",
        "year",
        "athletes",
        "age",
        "prev_medals",
        "medals"
    ]
]
```

The project then examines numerical correlations with the target variable:

```python
teams.select_dtypes(include=['number']).corr()['medals']
```

This helps identify which numerical variables have relationships with the number of medals won.

---

# 📈 Data Visualization

Several visualizations are generated to explore the dataset.

### Athletes vs Medals

```python
sns.lmplot(
    x="athletes",
    y="medals",
    data=teams,
    fit_reg=True,
    ci=None
)
```

This visualization examines whether teams with larger numbers of athletes tend to win more medals.

### Age vs Medals

```python
sns.lmplot(
    x="age",
    y="medals",
    data=teams,
    fit_reg=True,
    ci=None
)
```

This explores the relationship between average athlete age and medal performance.

### Medal Distribution

```python
teams.plot.hist(y="medals")
```

This displays the distribution of medal counts across the dataset.

---

# 🧹 Data Cleaning

Before training the model, missing values are identified:

```python
teams[teams.isnull().any(axis=1)]
```

Rows containing missing values are then removed:

```python
teams = teams.dropna()
```

This ensures that the selected features contain valid values before they are passed to the machine learning model.

---

# 🧪 Train/Test Split

Instead of randomly splitting the dataset, the project separates the data according to Olympic year.

### Training Data

```python
train = teams[teams["year"] < 2012].copy()
```

The model is trained using Olympic observations before 2012.

### Testing Data

```python
test = teams[teams["year"] >= 2012].copy()
```

The observations from 2012 onwards are used to evaluate the model.

This provides a more realistic setup because the model learns from earlier Olympic results and is then tested on later results.

---

# 🤖 Machine Learning Model

The project uses **Linear Regression** from Scikit-learn.

```python
from sklearn.linear_model import LinearRegression

reg = LinearRegression()
```

The model uses two predictors:

```python
predictors = [
    "athletes",
    "prev_medals"
]

target = "medals"
```

The model is trained using:

```python
reg.fit(
    train[predictors],
    train["medals"]
)
```

In simple terms, the model attempts to learn a mathematical relationship between:

* Number of athletes
* Previous medal performance
* Current medal performance

---

# 🔮 Making Predictions

Once the model has been trained, it predicts medal counts for the testing dataset:

```python
predictions = reg.predict(
    test[predictors]
)
```

The predictions are added to the testing dataset:

```python
test["predictions"] = predictions
```

Since the number of medals cannot be negative, negative predictions are replaced with zero:

```python
test.loc[
    test["predictions"] < 0,
    "predictions"
] = 0
```

The predictions are then rounded:

```python
test["predictions"] = test["predictions"].round()
```

---

# 📏 Model Evaluation

The project evaluates the model using **Mean Absolute Error (MAE)**.

```python
from sklearn.metrics import mean_absolute_error

error = mean_absolute_error(
    test["medals"],
    test["predictions"]
)
```

### What is MAE?

Mean Absolute Error measures the average absolute difference between the actual medal count and the predicted medal count.

For example:

```text
Actual medals:      20
Predicted medals:   17

Absolute error:      3
```

A **lower MAE indicates better predictions**.

---

# 🌍 Team-Level Error Analysis

The project also examines how prediction errors vary between teams.

First, the absolute error is calculated:

```python
errors = (
    test["medals"] -
    test["predictions"]
).abs()
```

The average error for each team is then calculated:

```python
error_by_team = errors.groupby(
    test["team"]
).mean()
```

The project also calculates average medals by team:

```python
medals_by_team = test["medals"].groupby(
    test["team"]
).mean()
```

This allows the project to investigate whether the model performs differently for teams with different levels of historical medal success.

---

# 📊 Error Ratio

An error ratio is calculated to provide another way of examining model performance:

```python
error_ratio = (
    error_by_team /
    medals_by_team
)
```

The project removes invalid values before visualizing the results:

```python
error_ratio = error_ratio[
    np.isfinite(error_ratio)
]
```

Finally, the distribution of the error ratios is visualized:

```python
error_ratio.plot.hist()

plt.show()
```

---

# 🛠️ Technologies Used

This project is built with Python and uses the following libraries:

* **Python**
* **Pandas** – Data loading and manipulation
* **NumPy** – Numerical operations
* **Matplotlib** – Data visualization
* **Seaborn** – Statistical visualization
* **Scikit-learn** – Machine learning and model evaluation

---

# 🚀 Getting Started

## 1. Clone the Repository

```bash
git clone https://github.com/JohnDakin/Athlete-winning-prediction-using-ML.git
```

## 2. Navigate to the Project

```bash
cd Athlete-winning-prediction-using-ML
```

## 3. Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## 4. Run the Project

```bash
python machine_learning.py
```

Make sure `teams.csv` is located in the same directory as `machine_learning.py`.

---

# 💡 How the Model Works

The model can be simplified as:

```text
              ┌─────────────────┐
              │ Number of       │
              │ Athletes        │
              └────────┬────────┘
                       │
                       │
                       ▼
              ┌─────────────────┐
              │                 │
              │ Linear          │
              │ Regression      │
              │ Model           │
              │                 │
              └────────┬────────┘
                       │
                       │
              ┌────────▼────────┐
              │ Predicted       │
              │ Medal Count     │
              └─────────────────┘
                       ▲
                       │
              ┌────────┴────────┐
              │ Previous       │
              │ Medals         │
              └─────────────────┘
```

The model learns from historical relationships between the number of athletes, previous medal performance, and actual medal counts.

---

# 🔬 Future Improvements

The current implementation provides a basic Linear Regression approach. There are several ways the project could be improved.

### Additional Features

More predictors could be added, including:

* Average athlete age
* Average height
* Average weight
* Number of events
* Previous three-year medal performance
* Historical country performance

### Alternative Machine Learning Models

The project could also compare Linear Regression with:

* Random Forest
* Gradient Boosting
* XGBoost
* Decision Trees
* Neural Networks

This would make it possible to determine whether more complex models provide better predictions.

### Improved Evaluation

Additional evaluation metrics could include:

* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* R² Score
* Mean Absolute Percentage Error (MAPE)

---

# 📚 Learning Objectives

This project demonstrates practical experience with:

* Loading and manipulating datasets with Pandas
* Exploratory Data Analysis
* Data cleaning
* Correlation analysis
* Data visualization
* Feature selection
* Training a Linear Regression model
* Making machine learning predictions
* Evaluating model performance
* Performing error analysis

---

# 👨‍💻 Author

**John Dakin**

GitHub: [JohnDakin](https://github.com/JohnDakin)

---

# 📖 Project Status

This project is an educational machine learning project created to explore the use of historical Olympic data for medal prediction.

The current implementation uses **Linear Regression** with `athletes` and `prev_medals` as the primary prediction features. The source code also identifies adding more predictors and experimenting with models such as Random Forests and neural networks as potential future improvements.

---

## ⭐ Acknowledgements

This project is intended for educational purposes and demonstrates how machine learning can be applied to sports and Olympic performance data.

If you find the project useful, consider giving the repository a ⭐.
