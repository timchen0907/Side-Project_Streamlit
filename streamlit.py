# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 11:10:10 2023

@author: 2211040
"""

#####################
# ---- Package ---- #
#####################
import time
import optuna
import warnings
import numpy as np 
import pandas as pd
import seaborn as sns
import lightgbm as lgb
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
warnings.filterwarnings("ignore")
st.set_option('deprecation.showPyplotGlobalUse', False)            

 
#######################
# ---- NBA Stats ---- #
#######################
# ---- Title ---- #
st.title('Welcome to my Streamlit')
st.latex(r'''
         Python + Streamlit = This Page!
         ''')
# ---- How to use streamlit ---- #
st.write("""
How to run ur streamlit
1. Open conda terminal
2. cd to ur_script.py folder
3. type python -m streamlit run ur_script.py folder

Reference: https://www.youtube.com/watch?v=JwSS70SZdyM&ab_channel=freeCodeCamp.org         
         """)
def main_page():
    # ---- Sidebar Header ---- #
    st.header('NBA Player Stats')
    # st.sidebar.markdown('<span style="font-size: 18px; font-weight: bold;">__Input Features__</span>', unsafe_allow_html=True)
    
    
    # ---- Year selection ---- #
    choose_year = st.sidebar.selectbox('Year', list(reversed(range(1997, 2024))))
    
    
    # ---- Scrape for NBA data ---- #
    @st.cache_data
    def load_data(year):
        url = "https://www.basketball-reference.com/leagues/NBA_" + str(year) + "_per_game.html"
        html = pd.read_html(url, header = 0)
        df = html[0]
        raw = df.drop(df[df.Age == 'Age'].index) # Deletes repeating headers in content
        raw = raw.fillna(0)
        playerstats = raw.drop(['Rk'], axis=1)
        return playerstats
    playerstats = load_data(choose_year)
    
    
    # ---- Order selection ---- #
    order_ls = ['PTS', 'AST', 'BLK', 'FG%']
    selected_order = st.sidebar.selectbox('Order', order_ls)
    
    
    # ---- Filter order ---- #
    selected_filter = st.sidebar.slider(selected_order, min_value = min(playerstats[selected_order].astype(float)), max_value = max(playerstats[selected_order].astype(float)))
    
    
    # ---- Sidebar - Team selection ---- #
    sorted_unique_team = sorted(playerstats.Tm.unique())
    selected_team = st.sidebar.multiselect('Team', sorted_unique_team, sorted_unique_team)
    if selected_team == []:
        st.error('Custom error: Pls choose at least a Team!')
        st.stop()
    # st.write(selected_team)
    
    
    # ---- Sidebar - Position selection ---- #
    unique_pos = ['C','PF','SF','PG','SG']
    selected_pos = st.sidebar.multiselect('Position', unique_pos, unique_pos)
    if selected_pos == []:
        st.error('Custom error: Pls choose at least a Position!')
        st.stop()
        
        
    # ---- Player stats ---- #
    df_selected_team = playerstats[(playerstats.Tm.isin(selected_team)) & (playerstats.Pos.isin(selected_pos))]
    find_df = df_selected_team.copy()
    df_selected_team[order_ls] = df_selected_team[order_ls].applymap(float)
    df_selected_team = df_selected_team[df_selected_team[selected_order] >= selected_filter]
    df_selected_team = df_selected_team.sort_values(selected_order, ascending = False)
    
    
    # ---- Best PTS, AST, 3P ---- #
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write('Most PTS:')
        pts = df_selected_team[df_selected_team['PTS']==df_selected_team['PTS'].max()]
        st.text(pts['Player'].values[0]+': '+str(pts['PTS'].values[0]))
    with col2:
        st.write('Most AST:')
        ast = df_selected_team[df_selected_team['AST']==df_selected_team['AST'].max()]
        st.text(ast['Player'].values[0]+': '+str(ast['AST'].values[0]))
    with col3:
        st.write('Most 3P:')
        ppp = df_selected_team[df_selected_team['3P']==df_selected_team['3P'].max()]
        st.text(ppp['Player'].values[0]+': '+str(ppp['3P'].values[0]))
    
    
    # ---- Show dataframe ---- #
    st.subheader('Player Stats')
    st.write('Data Dimension: ' + str(df_selected_team.shape[0]) + ' rows and ' + str(df_selected_team.shape[1]) + ' columns.')
    st.dataframe(df_selected_team)
    
    
    # ---- Reference Link ----#
    st.markdown("[Go to NBA Player Stats Reference](https://www.basketball-reference.com/leagues/NBA_2023_per_game.html)")
    
    
    # ---- Barplot analysis position's average score ---- #
    avg_pts = pd.DataFrame(df_selected_team.groupby('Pos')['PTS'].agg(['mean', 'count'])).reset_index()
    if st.toggle('Activate Bar'):
        st.header('Player Position Average Score')
        plt.figure(figsize=(10, 6))    
        bars = plt.bar(avg_pts['Pos'], avg_pts['mean'], color='orange')
        
        for bar, value in zip(bars, avg_pts.index):
            plt.text(bar.get_x() + bar.get_width() / 2, avg_pts['mean'][value] , str(round(avg_pts['mean'][value],2)), ha='center', va='bottom', fontsize=12)
            plt.text(bar.get_x() + bar.get_width() / 2, 2.5 , 'cnt: ' + str(avg_pts['count'][value]), ha='center', va='bottom', fontsize=12)
        
        plt.xticks(fontsize = 14)
        plt.yticks(fontsize = 14)
        plt.xlabel('Position', fontsize = 18)
        plt.ylabel('Average PTS', fontsize = 18)
        plt.grid(axis = 'y')
        st.pyplot()
    
    
    # ---- Find Favorite Player ---- #
    st.subheader('Find ur Favorite Player')
    player = st.text_input("Type Player's Full name (e.g.LeBron James)")
    find_result = find_df[find_df['Player'].str.lower()==player.lower()]
    search = st.button('Search')
    if search:
        if find_result.empty:
            st.write("Can't find ur favorite player, pls type it in correct spelling")
        else:
            st.dataframe(find_result)
            st.balloons()
            
# useful function: session_state

##############################
# ---- Machine Learning ---- #
##############################
def model_evaluation(model, X_train, y_train, X_test, y_test):
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    st.write("------------------Training Result--------------------")
    st.write("Training_Accuracy = {}".format(round(accuracy_score(y_train, train_pred), 4)))
    st.write("Training_Precision = {}".format(round(precision_score(y_train, train_pred), 4)))
    st.write("Training_Recall = {}".format(round(recall_score(y_train, train_pred), 4)))
    st.write("Training_F1-score = {}".format(round(f1_score(y_train, train_pred), 4)))
    
    st.write("------------------Testing Result--------------------")
    st.write("Testing_Accuracy = {}".format(round(accuracy_score(y_test, test_pred), 4)))
    st.write("Testing_Precision = {}".format(round(precision_score(y_test, test_pred), 4)))
    st.write("Testing_Recall = {}".format(round(recall_score(y_test, test_pred), 4)))
    st.write("Testing_F1-score = {}".format(round(f1_score(y_test, test_pred), 4)))
    
    
    return model
def second_page():
    st.title('Airline Passenger Satisfaction')
    
    # ---- LGBM Classifier Parameters ---- #
    lr = st.sidebar.slider('learning rate', value = 0.1, min_value=0.0, max_value= 1.0)
    n_estimator = st.sidebar.slider('n estimator', value = 100, min_value=1, max_value= 1000)
    l1 = st.sidebar.slider('reg alpha', value = 0.0, min_value=0.0, max_value= 1.0)
    l2 = st.sidebar.slider('reg lambda', value = 0.0, min_value=0.0, max_value= 1.0)
    lfs = st.sidebar.slider('num leaves', value = 31, min_value=1, max_value= 1000)
    colsample = st.sidebar.slider('colsample bytree', value = 1.0, min_value=0.0, max_value= 1.0)
    subsample = st.sidebar.slider('subsample', value = 1.0, min_value=0.0, max_value= 1.0)
    child = st.sidebar.slider('min child sample', value = 20, min_value=0, max_value= 100)
    
    # ---- training data ---- #
    st.subheader('Training Data Dimension')
    st.write('103904 rows and 24 columns.')
    st.dataframe(pd.read_csv('train_example.csv', encoding = 'utf-8-sig', usecols=range(1,25)))
    
    # ---- Data Processing ---- #
    st.subheader('Data Processing')
    st.write(
        '''
        Transform columns with below methods
        1. fill na with mean : Arrival Delay in Minutes
        2. one hot encoding : Gender, Customer Type, Type of Travel, satisfaction
        3. label encoding: Class
        '''
        )
    if st.toggle('Data Process Code'):
        code = '''
        # ---- fill na with mean ---- #
        train["Arrival Delay in Minutes"]=train["Arrival Delay in Minutes"].fillna(np.mean(train["Arrival Delay in Minutes"])) # 15.1787
        test["Arrival Delay in Minutes"]=test["Arrival Delay in Minutes"].fillna(np.mean(test["Arrival Delay in Minutes"])) # 14.7409
        
        # ---- one hot encoding ---- #
        train['Gender'].unique()
        class_mapping = { 'Male':1, 'Female':0}
        train['Gender'] = train['Gender'].map(class_mapping)
        test['Gender'] = test['Gender'].map(class_mapping)
        # 
        train['Customer Type'].unique()
        class_mapping = { 'Loyal Customer':1, 'disloyal Customer':0}
        train['Customer Type'] = train['Customer Type'].map(class_mapping)
        test['Customer Type'] = test['Customer Type'].map(class_mapping)
        #
        train['Type of Travel'].unique()
        class_mapping = { 'Personal Travel':0, 'Business travel':1}
        train['Type of Travel'] = train['Type of Travel'].map(class_mapping)
        test['Type of Travel'] = test['Type of Travel'].map(class_mapping)
        # 
        train['satisfaction'].unique()
        class_mapping = { 'neutral or dissatisfied':0, 'satisfied':1}
        train['satisfaction'] = train['satisfaction'].map(class_mapping)
        test['satisfaction'] = test['satisfaction'].map(class_mapping)
        
        # ---- label encoding ---- #
        train['Class'].unique()
        class_mapping = { 'Eco Plus':2, 'Business':3,'Eco':1}
        train['Class'] = train['Class'].map(class_mapping)
        test['Class'] = test['Class'].map(class_mapping)
        
        # ---- data split ---- #
        X_train = train.drop(columns="satisfaction")
        y_train =  train.satisfaction
        X_test = test.drop(columns = "satisfaction")
        y_test  = test.satisfaction
        '''
        st.code(code, language = 'python')
    st.write('')
    
    
    # ---- training ---- #
    st.subheader('Performance with default parameters')
    st.write('''
             ------------------Training Result--------------------\n
             Training_Accuracy = 0.9661\n
             Training_Precision = 0.9775\n
             Training_Recall = 0.9436\n
             Training_F1-score = 0.9602\n
             ------------------Testing Result--------------------\n
             Testing_Accuracy = 0.9642\n
             Testing_Precision = 0.9743\n
             Testing_Recall = 0.9433\n
             Testing_F1-score = 0.9586\n
             經過了1.511秒
             ''')
    if st.toggle('Training Code with default parameters'):
        code1 = '''
        def model_evaluation(model, X_train, y_train, X_test, y_test):
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            print("------------------Training Result--------------------")
            print("Training_Accuracy = {}".format(round(accuracy_score(y_train, train_pred),4)))
            print("Training_Precision = {}".format(round(precision_score(y_train, train_pred),4)))
            print("Training_Recall = {}".format(round(recall_score(y_train, train_pred), 4)))
            print("Training_F1-score = {}".format(round(f1_score(y_train, train_pred), 4)))
            
            print("------------------Testing Result--------------------")
            print("Testing_Accuracy = {}".format(round(accuracy_score(y_test, test_pred), 4)))
            print("Testing_Precision = {}".format(round(precision_score(y_test, test_pred), 4)))
            print("Testing_Recall = {}".format(round(recall_score(y_test, test_pred), 4)))
            print("Testing_F1-score = {}".format(round(f1_score(y_test, test_pred), 4)))
        start = time.time()
        clf = lgb.LGBMClassifier(random_state = 530)
        clf.fit(X_train, y_train)
        end = time.time()
        print("經過了{}".format(end-start),"秒")
        model_evaluation(clf,X_train, y_train, X_test, y_test)
        '''
        st.code(code1, language = 'python')
    st.write('')
    
    st.subheader('Performance with Optimization(Optuna)')
    st.write(
        '''
        ------------------Training Result--------------------\n
        Training_Accuracy = 0.9906\n
        Training_Precision = 0.9957\n
        Training_Recall = 0.9825\n
        Training_F1-score = 0.9891\n
        ------------------Testing Result--------------------\n
        Testing_Accuracy = 0.9663\n
        Testing_Precision = 0.9762\n
        Testing_Recall = 0.9462\n
        Testing_F1-score = 0.9610\n
        經過了2.8476650714874268 秒
        '''
        )
    
    if st.toggle('Training Code with Optimization(Optuna)'):
        code2 = '''
        start = time.time()
        clf_optuna = lgb.LGBMClassifier(learning_rate=0.07235046840654702,n_estimators=389,reg_alpha=0.22442622540074775,reg_lambda=0.4152448093239704,
        num_leaves=86,colsample_bytree=0.7531772141117373,subsample= 0.645426949332951,min_child_samples= 18,random_state=530)
        clf_optuna.fit(X_train, y_train)
        end = time.time()
        print("經過了{}".format(end-start),"秒")
        model_evaluation(clf_optuna,X_train, y_train, X_test, y_test)
        '''
        st.code(code2, language = 'python')
    st.write('')
    
    st.subheader('Try to beat above perfromance~')
    try_button = st.button('Setting OK, Try')
    if try_button:
        if 'training_process' not in st.session_state:
            train = pd.read_csv('./train.csv')
            train = train[[i for i in train.columns if "Unnamed" not in i]]
            test = pd.read_csv('./test.csv')
            test = test[[i for i in test.columns if "Unnamed" not in i]]
            train = train.drop('id', axis=1)
            test = test.drop('id', axis=1)
            # ---- fill na with mean ---- #
            train["Arrival Delay in Minutes"]=train["Arrival Delay in Minutes"].fillna(np.mean(train["Arrival Delay in Minutes"])) # 15.1787
            test["Arrival Delay in Minutes"]=test["Arrival Delay in Minutes"].fillna(np.mean(test["Arrival Delay in Minutes"])) # 14.7409
            
            # ---- one hot encoding ---- #
            train['Gender'].unique()
            class_mapping = { 'Male':1, 'Female':0}
            train['Gender'] = train['Gender'].map(class_mapping)
            test['Gender'] = test['Gender'].map(class_mapping)
            # 
            train['Customer Type'].unique()
            class_mapping = { 'Loyal Customer':1, 'disloyal Customer':0}
            train['Customer Type'] = train['Customer Type'].map(class_mapping)
            test['Customer Type'] = test['Customer Type'].map(class_mapping)
            #
            train['Type of Travel'].unique()
            class_mapping = { 'Personal Travel':0, 'Business travel':1}
            train['Type of Travel'] = train['Type of Travel'].map(class_mapping)
            test['Type of Travel'] = test['Type of Travel'].map(class_mapping)
            # 
            train['satisfaction'].unique()
            class_mapping = { 'neutral or dissatisfied':0, 'satisfied':1}
            train['satisfaction'] = train['satisfaction'].map(class_mapping)
            test['satisfaction'] = test['satisfaction'].map(class_mapping)
            
            # ---- label encoding ---- #
            train['Class'].unique()
            class_mapping = { 'Eco Plus':2, 'Business':3,'Eco':1}
            train['Class'] = train['Class'].map(class_mapping)
            test['Class'] = test['Class'].map(class_mapping)
        
            X_train = train.drop(columns="satisfaction")
            y_train =  train.satisfaction
            X_test = test.drop(columns = "satisfaction")
            y_test  = test.satisfaction
            st.session_state['training_process'] = 1
            st.session_state['X_train'] = X_train
            st.session_state['y_train'] = y_train
            st.session_state['X_test'] = X_test
            st.session_state['y_test'] = y_test
            
        clf_cus = lgb.LGBMClassifier(learning_rate = lr, 
                                     n_estimators = n_estimator,
                                     reg_alpha = l1,
                                     reg_lambda = l2,
                                     num_leaves = lfs,
                                     colsample_bytree = colsample,
                                     subsample = subsample,
                                     min_child_samples = child,
                                     random_state=530)
        start = time.time()
        clf_cus.fit(st.session_state['X_train'], st.session_state['y_train'])
        train_pred = clf_cus.predict(st.session_state['X_train'])
        test_pred = clf_cus.predict(st.session_state['X_test'])
        end = time.time()
        if accuracy_score(st.session_state['y_train'], train_pred) >  0.9906 and accuracy_score(st.session_state['y_test'], test_pred) < 0.9663:
            st.markdown('<span style="font-size: 18px; font-weight: bold;">Congrats!! Ur training result is better than mine! Keep trying to beat Testing result!</span>', unsafe_allow_html=True)
            st.write('')
            
        elif accuracy_score(st.seesion_state['y_test'], test_pred) > 0.9906:
            st.markdown('<span style="font-size: 18px; font-weight: bold;">Congrats!! Ur Testing result is better than mine! If training result works fine, this model is perfect!</span>', unsafe_allow_html=True)
            st.write('')
            
        elif accuracy_score(st.session_state['y_train'], train_pred) >  0.9906 and accuracy_score(st.seesion_state['y_test'], test_pred) > 0.9663:
            st.markdown('<span style="font-size: 18px; font-weight: bold;">Perfect Optimization!!!</span>', unsafe_allow_html=True)
            st.write('')
            
        else:
            st.markdown('<span style="font-size: 18px; font-weight: bold;">Keep Trying, u can do it!</span>', unsafe_allow_html=True)
            st.write('')
            
        st.write(model_evaluation(clf_cus, st.session_state['X_train'], st.session_state['y_train'],  st.session_state['X_test'], st.session_state['y_test']))
        st.write("經過了{}".format(end-start),"秒")


###########################
# ---- setting pages ---- #
###########################
st.sidebar.markdown('<span style="font-size: 24px; font-weight: bold;">__Inputs__</span>', unsafe_allow_html=True)
page = st.sidebar.selectbox('Selection', ['NBA', 'ML'])
if page == 'NBA':
    main_page()
elif page == 'ML':
    second_page()



 
# # download
# def filedownload(df):
#     csv = df.to_csv(index=False)
#     b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
#     href = f'<a href="data:file/csv;base64,{b64}" download="playerstats.csv">Download CSV File</a>'
#     return href

# st.markdown(filedownload(df_selected_team), unsafe_allow_html=True)

# # Heatmap
# if st.button('Intercorrelation Heatmap'):
#     st.header('Intercorrelation Matrix Heatmap')
#     df_selected_team.to_csv('output.csv',index=False)
#     df = pd.read_csv('output.csv')

#     corr = df.corr()
#     mask = np.zeros_like(corr)
#     mask[np.triu_indices_from(mask)] = True
#     with sns.axes_style("white"):
#         f, ax = plt.subplots(figsize=(7, 5))
#         ax = sns.heatmap(corr, mask=mask, vmax=1, square=True)
#     st.pyplot()
