## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.

STEP 2:Clean the Data Set using Data Cleaning Process.

STEP 3:Apply Feature Encoding for the feature in the data set.

STEP 4:Apply Feature Transformation for the feature in the data set.

STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
~~~
import pandas as pd
df=pd.read_csv('/content/Encoding Data.csv')
df
~~~
![image](https://github.com/user-attachments/assets/7ff1babc-c9ce-4b9d-9811-13941a4699e0)
# OrdinalEncoder
~~~
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
~~~
![image](https://github.com/user-attachments/assets/9f513685-14c4-4c4e-9d9a-081bdb2ceca3)
~~~
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
~~~
![image](https://github.com/user-attachments/assets/c137b9d6-0dc3-42a7-9eae-cdef60be98bc)
# LabelEncoder
~~~
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(df[["ord_2"]])
dfc
~~~
![image](https://github.com/user-attachments/assets/4077488e-0e07-4e12-bcde-9523b2fda1e6)
# OneHotEncoder
~~~
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[['nom_0']]))
df2=pd.concat([df2,enc],axis=1)
df2
~~~
![image](https://github.com/user-attachments/assets/27ae0114-58ee-4e4c-a4fe-02894be880f8)
~~~
pd.get_dummies(df2,columns=["nom_0"])
~~~
![image](https://github.com/user-attachments/assets/97fd5904-482d-4751-8ef1-d8c185968fba)
# BinaryEncoder
~~~
pip install --upgrade category_encoders
~~~
![image](https://github.com/user-attachments/assets/22cbe50a-a233-4424-91e9-2ea96ecf4944)
~~~
from category_encoders import BinaryEncoder
df=pd.read_csv('/content/data.csv')
df
~~~
![image](https://github.com/user-attachments/assets/0c82d130-0eac-44f3-9571-0a85556c34f9)
~~~
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb
~~~
![image](https://github.com/user-attachments/assets/94c9df83-5c1e-422c-9dc2-fab0ce524a48)
# TargetEncoder
~~~
from category_encoders import TargetEncoder
te=TargetEncoder()
cc=df.copy()
new=te.fit_transform(X=cc["City"],y=cc["Target"])
cc=pd.concat([cc,new],axis=1)
cc
~~~
![image](https://github.com/user-attachments/assets/ce1a4461-87b4-4f1a-b3db-1db8d0c28d6e)
# Data Transformation
~~~
import pandas as pd
import numpy as np
from scipy import stats
df=pd.read_csv('/content/Data_to_Transform.csv')
df
~~~
![image](https://github.com/user-attachments/assets/94022e99-13b0-4229-be99-d3851c47ae18)
~~~
df.skew()
~~~
![image](https://github.com/user-attachments/assets/c98ab1ed-14bb-4783-aca0-302b9ccf5cea)
~~~
np.log(df["Highly Positive Skew"])
~~~
![image](https://github.com/user-attachments/assets/95b621e7-d78b-46ba-b3f5-d7552a62efdb)
~~~
np.reciprocal(df["Moderate Positive Skew"])
~~~
![image](https://github.com/user-attachments/assets/622c7819-627b-4ec7-bdf2-5355f40a9076)
~~~
np.sqrt(df["Highly Positive Skew"])
~~~
![image](https://github.com/user-attachments/assets/50613c58-9c2c-4a4a-afd5-9c57b079fdbc)
~~~
np.square(df["Highly Positive Skew"])
~~~
![image](https://github.com/user-attachments/assets/214f0e54-288a-4789-9535-e6c2da9a5476)
~~~
df["Highly Positive Skew_boxcox"],parameters=stats.boxcox(df["Highly Positive Skew"])
df
~~~
![image](https://github.com/user-attachments/assets/57ce8efe-49f3-4542-952b-21b7a674e18b)
~~~
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
~~~
![image](https://github.com/user-attachments/assets/a3b179cd-edcc-4cae-b1a1-20cc99e7d891)
~~~
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
~~~
![image](https://github.com/user-attachments/assets/4e39f1d4-aee4-4db6-af9c-df88fecbb1b0)
~~~
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
~~~
![image](https://github.com/user-attachments/assets/3cc07389-9907-45e0-b621-172f2ecb745c)


# RESULT:
Finally,perform Feature Encoding and Transformation process is executed successfully.

       
