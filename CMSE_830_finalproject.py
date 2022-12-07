import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import neighbors
from sklearn.pipeline import make_pipeline
from sklearn import svm
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import streamlit as st
from PIL import Image
        

data =pd.read_csv("Placement_Data_Full_Class.csv")

X1 = data.drop(["sl_no","status","salary","etest_p","specialisation","mba_p","gender"],axis=1)
y = data["status"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(gamma=0.001, C=1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
]

#for i in classifiers:
#    i.fit(X_train, y_train)
#    print(i.score(X_test,y_test))

X = pd.get_dummies(X1,drop_first=True)
y = pd.get_dummies(y,drop_first=True)

start_state = 0 #random state is a model hyperparameter used to control the randomness involved in machine learning models
test_fraction = 0.4 # split fraction of the data into 20% testing and 80% training.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_fraction, random_state=start_state)
# splitting the data

my_scaler = StandardScaler()
# initialising the scaler

my_scaler.fit(X_train) # why isn't y_train used here? 
# we need to scale only the training data as it is our predictor variable.

X_train_scaled = my_scaler.transform(X_train)
# Scaling the X train data by calculating the mean and standard deviation and normalising the data

X_test_scaled = my_scaler.transform(X_test)

my_scaler.fit(X)
X_scaled = my_scaler.transform(X)

st.sidebar.title("Job Placement Prediction Model")
option = st.sidebar.radio(
    "What would you like to do",
    ('Dataset & EDA','Cross Validation','Hyperparameter Tuning','Models and Accuracies','Prediction Model'))

if option == 'Prediction Model':

    st.write("## Enter your details to see if  you can get placed")
    sbe = st.selectbox("Secondary School Board of Education", ("Central","Others"))
    sbep = st.slider('Secondary School percentage',0.0, 100.0)
    hbe = st.selectbox("HighSchool Board of Education", ("Central","Others"))
    s = st.selectbox("Highschool Stream", ("Commerce","Science","Arts"))
    hbep = st.slider('Highschool percentage',0.0, 100.0)
    d = st.selectbox("Degree Specialisation", ("Comm&Mgmt","Sci&Tech","Others"))
    dp = st.slider('Degree percentage',0.0, 100.0)
    wk = st.selectbox("Any Work Experience", ("Yes","No"))

    my_classifier3 = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5,algorithm='brute',weights='uniform'))
    my_model3 = my_classifier3.fit(X_train_scaled, y_train)

    df = pd.DataFrame({"ssc_p":sbep,"ssc_b":sbe,"hsc_p":hbep,"hsc_b":hbe,"hsc_s":s,
                        "degree_p":dp,
                         "degree_t":d,"workex":wk,
                         },index=[0])
    dummy = X1
    dummy = dummy.append(df,ignore_index = True)
    dummy = pd.get_dummies(dummy,drop_first = True)
    dummy = my_scaler.fit_transform(dummy)
    out = my_model3.predict(dummy[[-1]])
    if st.button('Show Status'):
        if out == 1:
         st.success('Congratulations you have high chancesof getting placeed 🎉', icon="🎉")
         st.balloons()
         #st.image('image.png')
        else:
            st.error('Better Luck Next time ', icon="⚠️")

elif option == "Dataset & EDA":
    st.title("Job Placement Prediction")
    image = Image.open("successful-placements-min.png")
    st.image(image,caption='image source from firefish blog') 
    st.header("Job Placement Dataset")
    st.dataframe(data)
    st.write("Data set is obtained from Kaggle.")
    nul = X1.isna().sum().sum()
    st.write("Null Values in Dataset")
    st.write(nul)
    fig0 = plt.figure(figsize=(10,5))
    sns.heatmap(data.corr(), annot= True)
    st.title('Exploratory Data Analysis')
    st.write("## Correlation Matrix")
    st.write(fig0)

    fig1 =plt.figure(figsize=(10,5))
    sns.histplot(data = data, x="degree_t")
    st.write("## Histplot")
    st.pyplot(fig1)

    fig2 =plt.figure(figsize=(10,5))
    sns.histplot(data = data, x="salary",hue='degree_t')
    st.pyplot(fig2)
    
    
elif option =="Models and Accuracies":
    imag = Image.open("E:\MSU\Classes\CMSE_830\images.jpg")
    st.title("Lets check the accuracies of different Classifiers")
    st.image(imag,caption='image source from bitesizebio.com') 
    model_c = st.radio("Select a classifier to see its accuracy",('KNN','Decision Tree','SVM','RandomForest'))
    if model_c == 'KNN':
        m1=KNeighborsClassifier(n_neighbors=5,algorithm='brute',weights='uniform').fit(X_train, y_train)
        st.success(m1.score(X_test,y_test))
        clff1 = make_pipeline(StandardScaler(), m1)
        moff1 = clff1.fit(X_train_scaled, y_train)
        cm1 = confusion_matrix(y,moff1.predict(X_scaled))
        fig0 = plt.figure(figsize=(10,5))
        sns.heatmap(cm1,annot=True,fmt='g')
        st.write("## Confusion Matrix")
        st.write(fig0)
    elif model_c == 'Decision Tree':
        m2=DecisionTreeClassifier(criterion='gini',max_depth=5,splitter='best',random_state = 50).fit(X_train, y_train)
        st.success(m2.score(X_test,y_test))
        clff2 = make_pipeline(StandardScaler(), m2)
        moff2 = clff2.fit(X_train_scaled, y_train)
        cm2 = confusion_matrix(y,moff2.predict(X_scaled))
        fig0 = plt.figure(figsize=(10,5))
        sns.heatmap(cm2,annot=True,fmt='g')
        st.write("## Confusion Matrix")
        st.write(fig0)
    elif model_c == 'SVM':
        m3=svm.SVC(gamma='auto',C=1,kernel ='linear').fit(X_train, y_train)
        st.success(m3.score(X_test,y_test))
        clff1 = make_pipeline(StandardScaler(), m3)
        moff1 = clff1.fit(X_train_scaled, y_train)
        cm1 = confusion_matrix(y,moff1.predict(X_scaled))
        fig0 = plt.figure(figsize=(10,5))
        sns.heatmap(cm1,annot=True,fmt='g')
        st.write("## Confusion Matrix")
        st.write(fig0)
    elif model_c == 'RandomForest':
        m4=RandomForestClassifier(max_depth=5, n_estimators=15, max_features=1,random_state = 1).fit(X_train, y_train)
        st.success(m4.score(X_test,y_test))
        clff1 = make_pipeline(StandardScaler(), m4)
        moff1 = clff1.fit(X_train_scaled, y_train)
        cm1 = confusion_matrix(y,moff1.predict(X_scaled))
        fig0 = plt.figure(figsize=(10,5))
        sns.heatmap(cm1,annot=True,fmt='g')
        st.write("## Confusion Matrix")
        st.write(fig0)

elif option =='Hyperparameter Tuning':
    imag = Image.open("manufacture.png")
    st.title("Hyper Parameter Tuning")
    st.image(imag,caption='image source from veryicon.com') 
    model_c = st.selectbox("Select a classifier to which are its best parameters",('KNN','Decision Tree','SVM','RandomForest'))
    if model_c == 'KNN':
        kn = GridSearchCV(KNeighborsClassifier(),{'weights':['uniform', 'distance'],'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute'],'n_neighbors':[1,3,5]},cv=5,return_train_score = False)
        kn.fit(X,y)
        st.write(kn.best_params_)
        st.write("Accuracy for the above prameters is:")
        st.write(kn.best_score_)
    elif model_c == 'Decision Tree':
        dt = GridSearchCV(DecisionTreeClassifier(random_state=50),{'criterion':['gini', 'entropy', 'log_loss'],'max_depth':[1,5,10],'splitter':['best','random']},cv=5,return_train_score = False)
        dt.fit(X,y)
        st.write(dt.best_params_)
        st.write("Accuracy for the above prameters is:")
        st.write(dt.best_score_)
    elif model_c == 'SVM':
        sv = GridSearchCV(svm.SVC(gamma= 'auto'),{'C':[1,10,20],'kernel':['rbf','linear']},cv=5,return_train_score = False)
        sv.fit(X,y)
        st.write(sv.best_params_)
        st.write("Accuracy for the above prameters is:")
        st.write(sv.best_score_)
    elif model_c == 'RandomForest':
        rf = GridSearchCV(RandomForestClassifier(random_state=1),{'max_depth':[1,5,10],'n_estimators':[1,5,10,15],'max_features':[1,2,3]},cv=5,return_train_score = False)
        rf.fit(X,y)
        st.write(rf.best_params_)
        st.write("Accuracy for the above prameters is:")
        st.write(rf.best_score_)

elif option =='Cross Validation':
    st.title("Cross Validation")
    k = st.slider("Select number of folds for KNN Classifier",2,5)
    cvk = cross_val_score(KNeighborsClassifier(n_neighbors=5,algorithm='brute',weights='uniform'),X,y,cv=k)
    st.write(cvk)
    st.write(np.mean(cvk))
    d = st.slider("Select number of folds for Decision Tree Classifier",2,5)
    cvd = cross_val_score(DecisionTreeClassifier(criterion='gini',max_depth=5,splitter='best',random_state=50),X,y,cv=d)
    st.write(cvd)
    st.write(np.mean(cvd))
    s = st.slider("Select number of folds for SVM classifier",2,5)
    cvs = cross_val_score(svm.SVC(gamma='auto',C=1,kernel ='linear'),X,y,cv=s)
    st.write(cvs)
    st.write(np.mean(cvs))
    r = st.slider("Select number of folds for Random Forest Classifier",2,5)
    cvr = cross_val_score(RandomForestClassifier(max_depth=5, n_estimators=15, max_features=1,random_state=1),X,y,cv=r)
    st.write(cvr)
    st.write(np.mean(cvr))









