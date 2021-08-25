#load requisite libraries
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import itertools
import matplotlib.gridspec as gridspec
import streamlit as st
import streamlit.components.v1 as stc

import re
#from sklearn.cross_validation import train_test_split
from sklearn import tree
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, NuSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import RFE, RFECV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
#from mlxtend.data import iris_data
from sklearn.utils import resample
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import streamlit.components.v1 as stc
import os

#stuff for plotting trees
from six import StringIO
#from sklearn.externals.six import StringIO from IPython.display import Image
from IPython.display import Image
from sklearn.tree import export_graphviz
#import pydotplus

st.set_option('deprecation.showPyplotGlobalUse', False)

def main():
    st.title("Machine Learning App for Classification")
    st.markdown ("The present supervised program requires a target class along with features")
    st.sidebar.title("Methods and Optimization")
     
    def load_data():
        global data
        data=[]
        global data_frame
        data_frame = pd.DataFrame()
        uploaded_file = st.file_uploader("Upload File",type=['csv'])
        if uploaded_file is not None:
            file_details = {"FileName":uploaded_file.name,"FileType":uploaded_file.type,"FileSize":uploaded_file.size}
            data_frame=pd.read_csv(uploaded_file,delimiter =',')
        data = data_frame.columns

        list=[]
        for i in range(0, len(data_frame.columns)):
            x=i
            list.append(x)
        global fixed_numbers
        global fixed_numbers_1
       
        fixed_numbers = st.sidebar.multiselect("Please select features", list)
        st.sidebar.text("maximum comparison is among 2 features")
        #st.write(fixed_numbers)
        st.sidebar.write("features in training:")
        
        for j in range (len(fixed_numbers)): 
            st.sidebar.write(data_frame.columns.values[int(fixed_numbers[j])]) 
            
        fixed_numbers_1 = st.sidebar.multiselect("Please select target class", list)
        #st.write(fixed_numbers)
        st.sidebar.write("target class for training:")
 
        for k in range(len(fixed_numbers_1)): 
            st.sidebar.write(data_frame.columns.values[int(fixed_numbers_1[k])])     
            
        labelencoder=LabelEncoder()
        for col in data_frame.columns:
            data_frame[col] = labelencoder.fit_transform(data_frame[col])
        return data_frame
    
    
    def split(df):
        global features
        global output
        features=[]
        output=[]
        #y = df.pathogenic
        #x = df.drop(columns=[[data_frame.iloc[:,fixed_numbers[0]],data_frame.iloc[:,fixed_numbers[1]],data_frame.iloc[:,fixed_numbers[2]]])
        features = [data[fixed_numbers[0]],data[fixed_numbers[1]]]
        output = [data[fixed_numbers_1[0]]]
        st.write(features, output)
        x = df[features]
        y = df[output]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
        return x_train, x_test, y_train, y_test

    def plot_metrics(metrics_list):
        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix")
            plot_confusion_matrix(model, x_test, y_test, display_labels=class_names)
            st.pyplot()

        if 'ROC Curve' in metrics_list:
            st.subheader("ROC Curve")
            plot_roc_curve(model, x_test, y_test)
            st.pyplot()
        
        if 'Precision-Recall Curve' in metrics_list:
            st.subheader('Precision-Recall Curve')
            plot_precision_recall_curve(model, x_test, y_test)
            st.pyplot()
        
        #if 'Overall Classification Metrics' in metrics_list:
         #   st.subheader("Overall Classification Metrics")
          #  classification_report(y_predict, y_test)
           # st.pyplot()

    df = load_data()
    
    class_names = ['Test', 'train']
    
    x_train, x_test, y_train, y_test = split(df)
    #y_predict = model.predict(x_test.values)

    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox("Classifier", ("Support Vector Machine (SVM)", "Logistic Regression", "Random Forest", "Decision Tree"))

    if classifier == 'Support Vector Machine (SVM)':
        st.sidebar.subheader("Model Hyperparameters")
        #choose parameters
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C_SVM')
        kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key='kernel')
        gamma = st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"), key='gamma')

        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))
        
        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Support Vector Machine (SVM) Results")
            model = SVC(C=C, kernel=kernel, gamma=gamma)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrics)
    
    if classifier == 'Logistic Regression':
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C_LR')
        max_iter = st.sidebar.slider("Maximum number of iterations", 100, 500, key='max_iter')

        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Logistic Regression Results")
            model = LogisticRegression(C=C, penalty='l2', max_iter=max_iter)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrics)
    
    if classifier == 'Random Forest':
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators = st.sidebar.number_input("The number of trees in the forest", 10, 50000, step=10, key='n_estimators')
        max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 20, step=1, key='n_estimators')
        bootstrap = st.sidebar.radio("Bootstrap samples when building trees", ('True', 'False'), key='bootstrap')
        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Random Forest Results")
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap, n_jobs=-1)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrics)

    if classifier == 'Decision Tree':
        st.sidebar.subheader("Model Hyperparameters")
        #n_estimators = st.sidebar.number_input("The number of trees in the forest", 10, 50000, step=10, key='n_estimators')
        max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 50, step=1, key='max_depth')
        #bootstrap = st.sidebar.radio("Bootstrap samples when building trees", ('True', 'False'), key='bootstrap')
        #metrics = st.sidebar.multiselect("What metrics to plot?", ('Overall Classification Metrics', 'Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))
        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Decision Tree Results")
            model = DecisionTreeClassifier(max_depth=max_depth)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrics)

    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Data Set")
        st.write(df)
    st.sidebar.markdown("Created by Kalyan Immadisetty")
        
if __name__ == '__main__':
    main()
