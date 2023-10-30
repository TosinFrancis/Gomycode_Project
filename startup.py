import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
import pickle
   
model = pickle.load(open('Exam_Performance_Model.pkl', 'rb'))
st.markdown("<h1 style = 'text-align: center; color: 3D0C11'>FACTORS THAT AFFECTS STUDENTS SCORE</h1> ", unsafe_allow_html = True)
st.markdown("<h6 style = 'top_margin: 0rem; text-align: center; color: #FFB4B4'>Built by OLANREWAJU HOPE</h6>", unsafe_allow_html = True)
st.image('hat.png', width = 400)


st.subheader('Project Brief')

st.markdown("<p style = 'top_margin: 0rem; text-align: justify; color: #FFB4B4', 'background-color:#002b36'> The data were obtained in a survey of students math and portuguese language courses in secondary school. It contains a lot of interesting social, gender and study information about students. You can use it for some EDA or try to predict students final grade</p>", unsafe_allow_html = True)

st.markdown("<br><br>", unsafe_allow_html = True)



username = st.text_input('Enter your name')
if st.button('submit name'):
    st.success(f"Welcome {username}. Pls use according to usage guidelines")

from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler = StandardScaler()
normalizing = MinMaxScaler()


dx = pd.read_csv('student-por.csv')

from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler = StandardScaler()
normalizing = MinMaxScaler()

for i in dx.columns:
    if dx[i].dtypes != 'O': # --------------- Select the numerical columns
        dx[[i]] = scaler.fit_transform(dx[[i]]) # ------------------------ Tranform the selected the numerical columns

from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder() # ---------------------------------------------------------- Instantiate encoding library
for i in dx.columns: # ------------------------------------------------------ iterate through the columns and find all categoricals
    if dx[i].dtypes == 'O': # ----------------------------------------------- select all categoricals
        dx[i] = lb.fit_transform(dx[i]) # -------------------------------- Transform the selected data.

dx.head()



heat = plt.figure(figsize = (14, 7))
sns.heatmap(dx.corr(), annot = True, cmap = 'BuPu')

st.write(heat)

data = pd.read_csv('student-por.csv')
st.write(data.sample(10))

df = pd.read_csv('student-por.csv')
target = pd.cut(x = df.G3, bins = [5,10,15,20], labels = ['Pass', 'Average', 'Excellent'])
df['Target'] = target
df.drop(['G1', 'G2', 'G3'], axis = 1, inplace = True)
df.head()
st.sidebar.image('exam.png', caption= f'Welcome {username}')

input_type = st.sidebar.selectbox('Select Your preffered Input type', ['Slider Input', 'Number Input'])


sel_col = ['school', 'higher', 'Dalc', 'Fedu', 'Walc', 'Medu', 'age']
df = df[sel_col]
if input_type == "Slider Input":
    age = st.sidebar.slider("age", df['age'].min(), df['age'].max())
    school = st.sidebar.select_slider("school", df['school'].unique())
    higher = st.sidebar.select_slider("higher", df['higher'].unique())
    Dalc = st.sidebar.slider("Dalc", df['Dalc'].min(), df['Dalc'].max())
    Walc = st.sidebar.slider("Walc", df['Walc'].min(), df['Walc'].max())
    Fedu = st.sidebar.slider("Fedu", df['Fedu'].min(), df['Fedu'].max())
    Medu = st.sidebar.slider("Medu", df['Medu'].min(), df['Medu'].max())

else:
    age = st.sidebar.number_input("age", df['age'].min(), df['age'].max())
    school = st.sidebar.number_input("school", df['age'].min(), df['age'].max())
    higher = st.sidebar.number_input("higher", df['age'].min(), df['age'].max())
    Dalc = st.sidebar.number_input("Dalc", df['Dalc'].min(), df['Dalc'].max())
    Walc = st.sidebar.number_input("Walc", df['Walc'].min(), df['Walc'].max())
    Fedu = st.sidebar.number_input("Fedu", df['Fedu'].min(), df['Fedu'].max())
    Medu = st.sidebar.number_input("Medu", df['Medu'].min(), df['Medu'].max())
  


 
input_variable = pd.DataFrame([{'school': school, 'higher': higher, 'Dalc': Dalc, 'Fedu': Fedu, 'Walc': Walc, 'Medu': Medu, 'age':age }])
st.write(input_variable)

for i in input_variable.columns:
    if input_variable[i].dtypes != 'O': # --------------- Select the numerical columns
        input_variable[[i]] = scaler.fit_transform(input_variable[[i]]) # ------------------------ Tranform the selected the numerical columns

from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder() # ---------------------------------------------------------- Instantiate encoding library
for i in input_variable.columns: # ------------------------------------------------------ iterate through the columns and find all categoricals
    if input_variable[i].dtypes == 'O': # ----------------------------------------------- select all categoricals
        input_variable[i] = lb.fit_transform(input_variable[i]) # -------------------------------- Transform the selected data.


pred_result, interpret = st.tabs(["Prediction Tab", "Interpretation Tab"])
with pred_result:
    if st.button('PREDICT'):

        st.markdown("<br>", unsafe_allow_html= True)
        prediction = model.predict(input_variable)
        st.write("Predicted Profit is :", prediction)
    else:
        st.write('Pls press the predict button for prediction')

with interpret:
    st.subheader('Model Interpretation')
    st.write(f"")

    st.markdown("<br>", unsafe_allow_html= True)

    st.markdown(f"- The expected Profit for a startup is ")

    st.markdown(f"- For every additional 1 dollar spent on R&D Spend, the expected profit is expected to increase by ")

    st.markdown(f"- For every additional 1 dollar spent on Administration Expense, the expected profit is expected to decrease by $ ")

    st.markdown(f"- For every additional 1 dollar spent on Marketting Expense, the expected profit is expected to increase by $")
