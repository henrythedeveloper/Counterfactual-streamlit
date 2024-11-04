```bash
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 10000 entries, 0 to 9999
Data columns (total 15 columns):
 #   Column                         Non-Null Count  Dtype
---  ------                         --------------  -----
 0   Stress_Level                   10000 non-null  object
 1   Sector                         10000 non-null  object
 2   Increased_Work_Hours           10000 non-null  int64
 3   Work_From_Home                 10000 non-null  int64
 4   Hours_Worked_Per_Day           10000 non-null  object
 5   Meetings_Per_Day               10000 non-null  object
 6   Productivity_Change            10000 non-null  int64
 7   Health_Issue                   10000 non-null  int64
 8   Job_Security                   10000 non-null  int64
 9   Childcare_Responsibilities     10000 non-null  int64
10  Commuting_Changes              10000 non-null  int64
11  Technology_Adaptation          10000 non-null  int64
12  Salary_Changes                 10000 non-null  int64
13  Team_Collaboration_Challenges  10000 non-null  int64
14  Affected_by_Covid              10000 non-null  int64
dtypes: int64(11), object(4)
memory usage: 1.1+ MB
```
# Observations:

    1. Columns with Object Data Type:
        - Stress_Level
        - Sector
        - Hours_Worked_Per_Day
        - Meetings_Per_Day

    2. Columns with Numeric Data Type (int64):
        - The remaining 11 columns are numeric.

# Next Steps: Data Preprocessing
Before building your model and integrating it with DiCE and Streamlit, it's important to preprocess your data properly. Let's address the columns with the object data type, as they may need special handling.
1. Examine Object Columns
a. Stress_Level

    Description: Likely represents the stress level of individuals.
    Action: Since this is your target variable, we need to understand its categories and encode it appropriately.

b. Sector

    Description: Represents the industry sector of the individual.
    Action: This is a categorical feature and should be encoded.

c. Hours_Worked_Per_Day and Meetings_Per_Day

    Description: These should logically be numeric features representing the number of hours worked per day and number of meetings per day.
    Observation: They are of type object, which suggests that they might contain non-numeric values or are stored as strings.
    Action: Convert these columns to numeric data types after handling any non-numeric entries.