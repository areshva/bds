import pandas as pd
import statsmodels.api as sm
import numpy as np
# Creating a dataset for the analysis
data = {
    'Condition': ['gain-urgent', 'neutral-urgent', 'loss-urgent', 'gain-noturgent', 'neutral-noturgent', 'loss-noturgent'],
    'Framing ': ['gain', 'neutral', 'loss', 'gain', 'neutral', 'loss'],
    'Urgency': ['urgent', 'urgent', 'urgent', 'not urgent', 'not urgent', 'not urgent'],
    'Detection Difficulty Numeric': [1, 3, 2, 1, 3, 2],  # Assuming numeric values for difficulty
    'Click Rate': [0.06531, 0.13027, 0.08319, 0.07453, 0.14286, 0.09724]
}

df = pd.DataFrame(data)
df['Detection Difficulty Numeric'] = pd.to_numeric(df['Detection Difficulty Numeric'], errors='coerce')
df['Click Rate'] = pd.to_numeric(df['Click Rate'], errors='coerce')

# Encoding the categorical data
df_encoded = pd.get_dummies(df, columns=['Framing ', 'Urgency'], drop_first=True)
import pandas as pd
import statsmodels.api as sm

# Assuming 'df' is your DataFrame
# Encoding the categorical data
df_encoded = pd.get_dummies(df, columns=['Framing ', 'Urgency'], drop_first=True)

# Defining the dependent variable (Click Rate) and independent variables
X = df_encoded.drop(['Click Rate', 'Condition'], axis=1)
y = df_encoded['Click Rate']

# Ensure X is all numeric
print(X.dtypes)  # Check to ensure all columns are numeric
X = X.select_dtypes(include=[np.number])  # Select only numeric columns

# Adding a constant to the model (intercept)
X = sm.add_constant(X)

# Performing Multiple Regression Analysis
model = sm.OLS(y, X).fit()

# Printing the summary of the regression
print(model.summary())
