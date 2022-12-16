import pandas as pd
from sklearn.linear_model import LinearRegression
# pip install pandas
# pip install numpy
# pip install scipy
# pip install scikit-learn


#create DataFrame
df = pd.DataFrame({'hours': [1, 2, 2, 4, 2, 1, 5, 4, 2, 4, 4, 3, 6],
                   'prep_exams': [1, 3, 3, 5, 2, 2, 1, 1, 0, 3, 4, 3, 2],
                   'score': [76, 78, 85, 88, 72, 69, 94, 94, 88, 92, 90, 75, 96]})

#view DataFrame
print(df)

#     hours  prep_exams  score
# 0       1           1     76
# 1       2           3     78
# 2       2           3     85
# 3       4           5     88
# 4       2           2     72
# 5       1           2     69
# 6       5           1     94
# 7       4           1     94
# 8       2           0     88
# 9       4           3     92
# 10      4           4     90
# 11      3           3     75
# 12      6           2     96

#initiate linear regression model
model = LinearRegression()

#define predictor and response variables
X, y = df[["hours", "prep_exams"]], df.score

#fit regression model
model.fit(X, y)

#calculate R-squared of regression model
r_squared = model.score(X, y)

#view R-squared value
print(r_squared)

# 0.7175541714105901