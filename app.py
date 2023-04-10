import streamlit as st
import pandas as pd
st.set_page_config(page_title = 'olvovchik_VKR', layout = "wide", initial_sidebar_state = 'expanded')
# st.title('Выпускная квалификационная работа по курсу «Data Science»')
st.markdown("<h1 style='text-align: center; color: gray;'>Выпускная квалификационная работа по курсу «Data Science»</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: black; font-weight: bold; font-size: 24px;'>Прогнозирование конечных свойств новых материалов (композиционных материалов)<p>", unsafe_allow_html=True)
st.markdown("<p style='text-align: right; color: black;'>Олейник Владимир Александрович</p>", unsafe_allow_html=True)

# @st.cache


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
    x9 = st.number_input('Потребление смолы, г/м2', min_value=0)
    x10 = st.number_input('Угол нашивки, град', min_value=0)
    x11 = st.number_input('Шаг нашивки', min_value=0)
    x11 = st.number_input('Плотность нашивки', min_value=0)

st.sidebar.header('Модели машинного обучения')
with st.sidebar:
    model = st.selectbox('Выберите модель', models_list)
    if st.button('Получить прогнозы', use_container_width = True):
        st.success('Успешно!', icon="✅")
        with col3:
            x1 = 0.3
            x8 = 70
            x9 = 2020
            # background-color: rgb(240, 242, 246)
            st.markdown("<p style='font-size: 14px;'>Соотношение матрица-наполнитель</p>", unsafe_allow_html=True)
            st.write(f"<p style='font-size: 20px; color: darkred;''>{x1}</p>", unsafe_allow_html=True) 
            st.write("<p style='font-size: 14px;'>Модуль упругости при растяжении, ГПа</p>", unsafe_allow_html=True)
            st.write(f"<p style='font-size: 20px; color: darkred;''>{x8}</p>", unsafe_allow_html=True)
            st.write("<p style='font-size: 14px;'>Прочность при растяжении, МПа</p>", unsafe_allow_html=True)
            st.write(f"<p style='font-size: 20px; color: darkred;''>{x9}</p>", unsafe_allow_html=True)
