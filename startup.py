import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import pickle
   

#model = pickle.load(open('Student_Performance_.pkl', 'rb'))
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


data = pd.read_csv('student-por.csv')

for i in data.columns:
    if data[i].dtypes == 'int' or data[i].dtypes == 'float': # --------------- Select the numerical columns
        data[[i]] = scaler.fit_transform(data[[i]]) # ------------------------ Tranform the selected the numerical columns

from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder() # ---------------------------------------------------------- Instantiate encoding library
for i in data.columns: # ------------------------------------------------------ iterate through the columns and find all categoricals
    if data[i].dtypes == 'O': # ----------------------------------------------- select all categoricals
        data[i] = lb.fit_transform(data[i]) # -------------------------------- Transform the selected data.

heat = plt.figure(figsize = (14, 7))
sns.heatmap(data.corr(), annot = True, cmap = 'BuPu')

st.write(heat)

df = pd.read_csv('student-por.csv')
st.write(df.sample(10))

st.sidebar.image('exam.png', caption= f'Welcome {username}')

input_type = st.sidebar.selectbox('Select Your preffered Input type', ['Slider Input', 'Number Input'])


sel_col = ['school', 'higher', 'Dalc', 'Fedu', 'Walc', 'age', 'Medu', 'Mjob', 'failures', 'studytime', 'internet']
df = df[sel_col]
if input_type == "Slider Input":
    age = st.sidebar.slider("age", df['age'].min(), df['age'].max())
    studytime = st.sidebar.slider("studytime", df['studytime'].min(), df['studytime'].max())
    failures = st.sidebar.slider("failures", df['failures'].min(), df['failures'].max())
    school = st.sidebar.select_slider("school", df['school'].unique())
    higher = st.sidebar.select_slider("higher", df['higher'].unique())
    Dalc = st.sidebar.slider("Dalc", df['Dalc'].min(), df['Dalc'].max())
    Walc = st.sidebar.slider("Walc", df['Walc'].min(), df['Walc'].max())
    Fedu = st.sidebar.slider("Fedu", df['Fedu'].min(), df['Fedu'].max())
    Medu = st.sidebar.slider("Medu", df['Medu'].min(), df['Medu'].max())
    Mjob = st.sidebar.select_slider("Mjob", df['Mjob'].unique())
    internet = st.sidebar.select_slider("internet", df['internet'].unique())
else:
    age = st.sidebar.number_input("age", df['age'].min(), df['age'].max())
    failures = st.sidebar.number_input("failures", df['failures'].min(), df['failures'].max())
    studytime = st.sidebar.number_input("studytime", df['studytime'].min(), df['studytime'].max())
    Dalc = st.sidebar.number_input("Dalc", df['Dalc'].min(), df['Dalc'].max())
    Walc = st.sidebar.number_input("Walc", df['Walc'].min(), df['Walc'].max())
    Fedu = st.sidebar.number_input("Fedu", df['Fedu'].min(), df['Fedu'].max())
    Medu = st.sidebar.number_input("Medu", df['Medu'].min(), df['Medu'].max())
    
input_variable = pd.DataFrame([{'school': school, 'higher': higher, 'Dalc': Dalc, 'Fedu': Fedu, 'Walc': Walc, 'Medu': Medu, 'age':age, 'Mjob':Mjob, 'failures':failures, 'studytime':studytime, 'internet': internet }])
st.write(input_variable)

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
    st.write(f"Profit = {model.intercept_.round(2)} + {model.coef_[0].round(2)} R&D Spend + {model.coef_[1].round(2)} Administration + {model.coef_[2].round(2)} Marketing Spend")

    st.markdown("<br>", unsafe_allow_html= True)

    st.markdown(f"- The expected Profit for a startup is {model.intercept_}")

    st.markdown(f"- For every additional 1 dollar spent on R&D Spend, the expected profit is expected to increase by ${model.coef_[0].round(2)}  ")

    st.markdown(f"- For every additional 1 dollar spent on Administration Expense, the expected profit is expected to decrease by ${model.coef_[1].round(2)}  ")

    st.markdown(f"- For every additional 1 dollar spent on Marketting Expense, the expected profit is expected to increase by ${model.coef_[2].round(2)}  ")
