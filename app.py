import streamlit as st
import os
import pandas as pd
import joblib as jb

glowing_text_style = '''
    <style>
        .glowing-text {
            font-family: 'Arial Black', sans-serif;
            font-size: 33px;
            text-align: center;
            animation: glowing 2s infinite;
        }
        
        @keyframes glowing {
            0% { color: #FF9933; } /* Saffron color */
            10% { color: #FFD700; } /* Gold color */
            20% { color: #FF1493; } /* Deep Pink */
            30% { color: #00FF00; } /* Lime Green */
            40% { color: #FF4500; } /* Orange Red */
            50% { color: #9400D3; } /* Dark Violet */
            60% { color: #00BFFF; } /* Deep Sky Blue */
            70% { color: #FF69B4; } /* Hot Pink */
            80% { color: #ADFF2F; } /* Green Yellow */
            90% { color: #1E90FF; } /* Dodger Blue */
            100% { color: #FF9933; } /* Saffron color */
        }
    </style>
'''


st.markdown(glowing_text_style, unsafe_allow_html=True)
st.markdown(f'<p class="glowing-text">Heart Stroke Classification</p>', unsafe_allow_html=True)

def return_df(gender,age,hypertension,heart_disease,ever_married,work_type,Residence_type,avg_glucose_level,bmi,smoking_status):
	kbn={
	'gender':[gender],
	'age':[age],
	'hypertension':[hypertension],
	'heart_disease':[heart_disease],
	'ever_married':[ever_married],
	'work_type':[work_type],
    'Residence_type':[Residence_type],
    'avg_glucose_level':[avg_glucose_level],
    'bmi':[bmi],
    'smoking_status':[smoking_status]
	}
	final_df=pd.DataFrame(kbn)
	return final_df

def base_model():
    bmodel=jb.load(os.path.join('finalized_nb_model (1).pkl'))
    return bmodel
gender=st.selectbox('Select your gender',['Male','Female'])
age=st.number_input('Enter your age',min_value=0)
hypertension=st.slider('hypertension',0,1,0)
heart_disease=st.slider('heart_disease',0,1,0)
ever_married=st.selectbox('ever_married ?',['Yes','No'])
work_type=st.selectbox('work_type',['Private','Self-employed','Govt_job'])
Residence_type=st.selectbox('Residence',['Urban','Rural'])
avg_glucose_level=st.number_input('enter your glucose',min_value=0)
bmi=st.number_input('bmi',min_value=0)
smoking_status=st.selectbox('smoking_status',['formerly smoked','never smoked ','smokes','Unknown'])

df=return_df(gender,age,hypertension,heart_disease,ever_married,work_type,Residence_type,avg_glucose_level,bmi,smoking_status)
if st.button('Submit'):
   model=base_model()
   preds=model.predict(df)
   predictions=preds[0]
   if predictions==1:
      st.write('Stroke')
   elif predictions==0:
        st.write('Not Stroke')
        st.balloons()
