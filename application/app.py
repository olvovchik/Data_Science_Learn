import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from keras.models import load_model

st.set_page_config(page_title = 'olvovchik_VKR', layout = "wide", initial_sidebar_state = 'expanded')
st.markdown("<h1 style='text-align: center; color: gray;'>Выпускная квалификационная работа по курсу «Data Science»</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: black; font-weight: bold; font-size: 24px;'>Прогнозирование конечных свойств новых материалов (композиционных материалов)<p>", unsafe_allow_html=True)
st.markdown("<p style='text-align: right; color: black;'>Олейник Владимир Александрович</p>", unsafe_allow_html=True)


@st.cache_data
def get_dataset(X, target):
    if target in (1,2):
        columns_name = ['Плотность, кг/м3', 'Модуль упругости, ГПа', 'Количество отвердителя, м.%',
                        'Содержание эпоксидных групп,%_2', 'Температура вспышки, С_2',
                        'Поверхностная плотность, г/м2', 'Потребление смолы, г/м2',
                        'Угол нашивки, град', 'Шаг нашивки', 'Плотность нашивки']
    elif target == 3:
         columns_name = ['Плотность, кг/м3', 'Модуль упругости, ГПа', 'Количество отвердителя, м.%',
                        'Содержание эпоксидных групп,%_2', 'Температура вспышки, С_2',
                        'Поверхностная плотность, г/м2', 'Модуль упругости при растяжении, ГПа',
                        'Прочность при растяжении, МПа', 'Потребление смолы, г/м2',
                        'Угол нашивки, град', 'Шаг нашивки', 'Плотность нашивки']  
                  
    df = pd.DataFrame(np.array(X).reshape(1,-1), columns = columns_name)
    return df



@st.cache_data
def target_predict(predictor, target, X):  
    if predictor == 'LinearRegression':
        predict_name = 'LinReg'
    elif predictor == 'DecisionTreeRegressor':    
        predict_name = 'DecTreeReg'
    elif predictor == 'RandomForestRegressor':    
        predict_name = 'RandForReg'
    elif predictor == 'LinearSVR':    
        predict_name = 'LinSVR'
    elif predictor == 'KNeighborsRegressor':    
        predict_name = 'KNNReg'                

    if target == 1:
        target_name = 't1'
    elif target == 2:  
        target_name = 't2'

    file_path = Path(f'ModelML/{predict_name}_{target_name}.pkl')
    # file_path = (current_path+f'/ModelML/{predict_name}_{target_name}.pkl')
    regressor = pickle.load(open(file_path,'rb'))   
    y = regressor.predict(X)
    return y


@st.cache_data
def neuron_predict(X):  
    file_path = Path('ModelNN/preprocessorNN.pkl')
    preprocessor = pickle.load(open(file_path,'rb'))   
    X = preprocessor.transform(X)
    file_path = Path('ModelNN/Final_model.h5')
    regressor = load_model(file_path)
    y = regressor.predict(X)
    return y


models_list = ['LinearRegression', 
                'DecisionTreeRegressor',
                'RandomForestRegressor',
                'LinearSVR',
                'KNeighborsRegressor']


col1, col2, col3  = st.columns(3, gap = 'medium')


with col1:
    x2 = st.number_input('Плотность, кг/м3', min_value=0, )
    x3 = st.number_input('Модуль упругости, ГПа', min_value=0)
    x4 = st.number_input('Количество отвердителя, м.%', min_value=0)
    x5 = st.number_input('Содержание эпоксидных групп,%_2', min_value=0)
    x6 = st.number_input('Температура вспышки, С_2', min_value=0)


with col2:    
    x7 = st.number_input('Поверхностная плотность, г/м2', min_value=0)
    x10 = st.number_input('Потребление смолы, г/м2', min_value=0)
    x11 = st.selectbox('Угол нашивки, град', [0, 90])
    x12 = st.number_input('Шаг нашивки', min_value=0)
    x13= st.number_input('Плотность нашивки', min_value=0)


st.sidebar.header('Модели машинного обучения')
with st.sidebar:
    model = st.selectbox('Выберите модель', models_list)
    if st.button('Получить прогнозы', use_container_width = True):
        st.image('Image/IconCheck.png')
        # st.success('Успешно!', icon="✅")
        with col3:
            x8 = target_predict(model, 1, get_dataset([x2,x3,x4,x5,x6,x7,x10,x11,x12,x13],1))
            x9 = target_predict(model, 2, get_dataset([x2,x3,x4,x5,x6,x7,x10,x11,x12,x13],2))
            x1 = neuron_predict(get_dataset([x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13],3))
            st.markdown("<p style='font-size: 14px;'>Соотношение матрица-наполнитель</p>", unsafe_allow_html=True)
            st.write(f"<p style='font-size: 20px; color: darkred;''>{x1[0][0]:.4f}</p>", unsafe_allow_html=True) 
            st.write("<p style='font-size: 14px;'>Модуль упругости при растяжении, ГПа</p>", unsafe_allow_html=True)
            st.write(f"<p style='font-size: 20px; color: darkred;''>{x8[0]:.2f}</p>", unsafe_allow_html=True)
            st.write("<p style='font-size: 14px;'>Прочность при растяжении, МПа</p>", unsafe_allow_html=True)
            st.write(f"<p style='font-size: 20px; color: darkred;''>{x9[0]:.2f}</p>", unsafe_allow_html=True)
            st.image('Image/Neuron.png')
        with col1:
            st.image('Image/Target1.png', "Метрики 'Модуль упругости при растяжении, ГПа'")
        with col2:
            st.image('Image/Target2.png', "Метрики 'Прочность при растяжении, МПа'")    