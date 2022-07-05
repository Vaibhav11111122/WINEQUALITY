import pandas as pd
df = pd.read_csv('/winequalityN.csv')
print(df)
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt 
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB 
lrg = LinearRegression()
lr = LogisticRegression()
le = LabelEncoder()
gbc = GradientBoostingClassifier(n_estimators=10)
rf = RandomForestClassifier(random_state=1)
gnb = GaussianNB() 
wht = df[df['type'] == 'white'] 
print(wht)
newwht = (wht.isnull().sum())
print(newwht)
nwt = df.fillna(0)
x = nwt.drop('type',axis=1)
x = x.drop('quality',axis=1)
print(x)
y = nwt['quality']
df.boxplot()
plt.show()
**Logistic Regression**
X_train, X_test, Y_train, Y_test = train_test_split(x,y,random_state=0,test_size=0.3)
train = lr.fit(X_train, Y_train)
y_pred = lr.predict(X_test)
print(accuracy_score(Y_test, y_pred))
**Gradient boosting**
train_gbc =gbc.fit(X_train, Y_train)
gbc_pred = gbc.predict(X_test)
print(accuracy_score(Y_test, gbc_pred))
**Random Forest**
train_rf = rf.fit(X_train, Y_train)
rf_pred = rf.predict(X_test)
print(accuracy_score(Y_test, rf_pred))
**GaussionNB**
train_gnb = gnb.fit(X_train, Y_train)
gnb_pred = gnb.predict(X_test)
print(accuracy_score(Y_test, gnb_pred))










