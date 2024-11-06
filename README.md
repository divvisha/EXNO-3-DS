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
![Screenshot 2024-11-06 111251](https://github.com/user-attachments/assets/1f2bc63a-0f80-42ad-ba45-1206524f538a)
# OrdinalEncoder
~~~
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
~~~
![Screenshot 2024-11-06 111303](https://github.com/user-attachments/assets/f1e981a7-9b52-47d3-9bba-2dfa0d8b1ea6)
~~~
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
~~~
![Screenshot 2024-11-06 111346](https://github.com/user-attachments/assets/e06525c6-4cae-48f3-9722-5b7f4cbce845)
# LabelEncoder
~~~
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(df[["ord_2"]])
dfc
~~~
![Screenshot 2024-11-06 111356](https://github.com/user-attachments/assets/a1ea322c-cc5b-4ecc-83c4-9717472c3de5)
# OneHotEncoder
~~~
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[['nom_0']]))
df2=pd.concat([df2,enc],axis=1)
df2
~~~
![Screenshot 2024-11-06 111413](https://github.com/user-attachments/assets/7196171b-511e-40fb-a2a3-7ab2810b124c)
~~~
pd.get_dummies(df2,columns=["nom_0"])
~~~
![Screenshot 2024-11-06 111422](https://github.com/user-attachments/assets/95e5a667-6777-42ad-afaf-87f44a0537cf)
# BinaryEncoder
~~~
pip install --upgrade category_encoders
~~~
![Screenshot 2024-11-06 111438](https://github.com/user-attachments/assets/2139d4ea-1c66-431e-ba06-f41438033f61)
~~~
from category_encoders import BinaryEncoder
df=pd.read_csv('/content/data.csv')
df
~~~
![Screenshot 2024-11-06 111446](https://github.com/user-attachments/assets/32abeffe-e508-4aaa-b47e-f28fb0d6cb8b)
~~~
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb
~~~
![Screenshot 2024-11-06 111455](https://github.com/user-attachments/assets/c07a9067-a776-4f5b-bc94-464a55b4f4a3)
# TargetEncoder
~~~
from category_encoders import TargetEncoder
te=TargetEncoder()
cc=df.copy()
new=te.fit_transform(X=cc["City"],y=cc["Target"])
cc=pd.concat([cc,new],axis=1)
cc
~~~
![Screenshot 2024-11-06 111503](https://github.com/user-attachments/assets/49ab6c61-29b5-40ff-b417-496f5a111034)
# Data Transformation
~~~
import pandas as pd
import numpy as np
from scipy import stats
df=pd.read_csv('/content/Data_to_Transform.csv')
df
~~~
![Screenshot 2024-11-06 111511](https://github.com/user-attachments/assets/9f0119e7-20c5-43a9-b348-46cc6f4b46f6)
~~~
df.skew()
~~~
![Screenshot 2024-11-06 111528](https://github.com/user-attachments/assets/dba468ac-d243-43bf-aa81-a982f3045fed)
~~~
np.log(df["Highly Positive Skew"])
~~~
![Screenshot 2024-11-06 111541](https://github.com/user-attachments/assets/3e8dfe2d-bb1e-41aa-837e-88ab4faa8dd8)
~~~
np.reciprocal(df["Moderate Positive Skew"])
~~~
![Screenshot 2024-11-06 111549](https://github.com/user-attachments/assets/a91625fd-a9c7-4c41-9214-8f3a0c1f2f57)
~~~
np.sqrt(df["Highly Positive Skew"])
~~~
![Screenshot 2024-11-06 111558](https://github.com/user-attachments/assets/7257fd0e-8b55-4d5c-b491-e3730432affd)
~~~
np.square(df["Highly Positive Skew"])
~~~
![Screenshot 2024-11-06 111607](https://github.com/user-attachments/assets/75bb54ab-80cb-4c33-8694-c56accb02ee9)
~~~
df["Highly Positive Skew_boxcox"],parameters=stats.boxcox(df["Highly Positive Skew"])
df
~~~
![Screenshot 2024-11-06 111624](https://github.com/user-attachments/assets/80d97a72-d400-468f-8c7a-12f04b71bab5)
~~~
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
~~~
![Screenshot 2024-11-06 111633](https://github.com/user-attachments/assets/32558398-392b-4b14-ba90-ab1df6c9801f)
~~~
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
~~~
![Screenshot 2024-11-06 111642](https://github.com/user-attachments/assets/6e8387f5-b86f-4350-8d15-46ffa9acf9d3)
~~~
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
~~~
![Screenshot 2024-11-06 111707](https://github.com/user-attachments/assets/49161fd2-8efa-4829-9ac7-f4640562ca9e)


# RESULT:
Finally,perform Feature Encoding and Transformation process is executed successfully.

       
