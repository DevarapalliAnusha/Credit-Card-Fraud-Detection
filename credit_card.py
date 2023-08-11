import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
data=pd.read_excel(r'C:/Users/xyz python/creditcard.xlsx')
#print(data.info())
legit=data[data.Class==0]
fraud=data[data.Class==1]
#print(legit.shape)
#print(fraud.shape)
#print(legit.describe())
#print(fraud.describe())
#print(data.groupby('Class').mean())
legit_sample=legit.sample(n=492)
#print(legit_sample.head())
d1=pd.concat([legit_sample,fraud],axis='rows')
#print(d1.head)
x=d1.drop(['Class'],axis='columns')
y=d1.Class
#print(d1.groupby('Class').mean())
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=10)
#print(x_train.shape)
#print(x_test.shape)
scores=[]
model_params={'logisticregression':{'model':LogisticRegression(solver='liblinear',multi_class='auto'),'params':{'C':[10,30,40,50]}}}
k=LogisticRegression()
p=(cross_val_score(k,x,y,cv=5)).mean()
print(p)
#'svm':{'model':SVC(gamma='auto'),'params':{'C':[1,5,10],'kernel':['rbf','linear']}},
#'logisticregression':{'model':LogisticRegression(solver='liblinear',multi_class='auto'),'params':{'C':[1,10,20]}}
#'DecisionTreeClassifier':{'model':DecisionTreeClassifier(),'params':{'criterion':['gini','entropy']}},
#'randomforestclassifier':{'model':RandomForestClassifier(),'params':{'n_estimators':[1,10,20]}},
#'GaussianNB':{'model':GaussianNB(),'params':{}}}
for model_name,mp in model_params.items():
    dog=GridSearchCV(mp['model'],mp['params'],cv=5,return_train_score=False)
    dog.fit(x,y)
    scores.append({'model': model_name,'best_score':dog.best_score_,'best_params':dog.best_params_})
pat=pd.DataFrame(scores,columns=['model','best_score','best_params'])
print(pat)


