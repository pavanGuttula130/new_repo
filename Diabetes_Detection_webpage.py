import pickle
import numpy as np
import streamlit as st

#loading a files

def features_(input_):
    load_file=pickle.load(open('C:/Users/pavan/OneDrive/Documents/PYTHON/Diabetes_Detection/trained_model.sav','rb'))
    load_file1=pickle.load(open('C:/Users/pavan/OneDrive/Documents/PYTHON/Diabetes_Detection/StandardScaler.sav','rb'))
    list_to_array=np.asarray(input_)
    reshape_array=list_to_array.reshape(1,-1)
    standard_array=load_file1.transform(reshape_array)
    print(standard_array)
    test_predict=load_file.predict(standard_array)
    print(test_predict)
    if test_predict[0]==0:
        return 'Non diabetic'
    else:
        return 'Diabetic'
def main():
    st.title('Diabetes prediction web app')
    #Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome
    Pregnancies=st.text_input('Pregnancies')
    Glucose=st.text_input('Glucose content')
    BloodPressure=st.text_input('Blood pressure value')
    SkinThickness=st.text_input('Enter the value of Thickness of skin')
    Insulin=st.text_input('Insulin content')
    BMI=st.text_input('Enter the value of BMI')
    DiabetesPedigreeFunction=st.text_input('DiabetesPedigreeFunction')
    Age=st.text_input('Enter the age')

    #code for prediction
    diagnosis=' '
    if st.button('Diabetes Test result'):
        diagnosis=features_([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction,Age])

    st.success(diagnosis)
if __name__== '__main__':
    main()