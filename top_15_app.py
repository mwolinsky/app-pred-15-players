#!/usr/bin/env python
# coding: utf-8

# In[1]:

import xgboost as xgb
import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image
from numpy import loadtxt
from xgboost import XGBClassifier
import urllib.request
import streamlit.components.v1 as components
#import shap
import lime
from lime.lime_tabular import LimeTabularExplainer

model=xgb.XGBClassifier()

model.load_model("xgb_model.json")


data_location= "https://raw.githubusercontent.com/mwolinsky/Ranking_predictor/main/england-premier-league-players-2018-to-2019-stats.csv"

#Leemos el csv con lo jugadores como indice
df= pd.read_csv(data_location)

#Comenzamos con la creación de la variable target, cuyo valor serán los mejores 15 jugadores de cada posición

df['top_mid_15']=df.rank_in_league_top_midfielders.apply(lambda x: 1 if x>0 and x<=15 else 0)
df['top_def_15']=df.rank_in_league_top_defenders.apply(lambda x: 1 if x>0 and x<=15 else 0)
df['top_att_15']=df.rank_in_league_top_attackers.apply(lambda x: 1 if x>0 and x<=15 else 0)
df['top_15']= df.apply(lambda x: 1 if x.top_mid_15==1 or x.top_def_15==1 or x.top_att_15==1 else 0,axis=1)
df=df.drop(columns=['top_mid_15', 'top_att_15', 'top_def_15'])

df['top_rank']=df.loc[:,['rank_in_league_top_attackers','rank_in_league_top_midfielders','rank_in_league_top_defenders']].apply(lambda x: x.min(),axis=1)


categorical_columns=['position','Current Club','nationality']
for column in categorical_columns:
    dummies = pd.get_dummies(df[column], prefix=column,drop_first=True)
    df = pd.concat([df, dummies], axis=1)
    df = df.drop(columns=column)

X=df.loc[:,['goals_per_90_overall', 'assists_per_90_overall', 'goals_involved_per_90_overall', 'min_per_conceded_overall', 'minutes_played_overall']]
y= df.top_15

from sklearn.model_selection import train_test_split

#Con estratificación en y

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,random_state=162)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train),columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test),columns=X_test.columns)

X_train_explainer = np.array(X_train)
explainer = LimeTabularExplainer(X_train_explainer, 
                                 mode = "classification",
                                 training_labels = y_train,
                                 feature_names = X_train.columns, 
                                 categorical_features  = list(range(5)),
                                 discretize_continuous=False,
                                class_names=['Not Top 15','Top 15'])

@st.cache
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)
    
#def explainer(model):
    #explainer0 = shap.TreeExplainer(model)

    #shap_values0= ""
    #return(explainer0,shap_values0)
    
def st_shap(plot, height=None):
    print(type(shap))
    print(dir(shap))
    js=shap.getjs()
    shap_html = f"<head>{js}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)
    

    
def welcome(): 
    return 'welcome all'
  
def prediction(goals_per_90_overall, assists_per_90_overall, goals_involved_per_90_overall, min_per_conceded_overall, minutes_played_overall):   
    mw=np.array([goals_per_90_overall, assists_per_90_overall, goals_involved_per_90_overall, min_per_conceded_overall, minutes_played_overall]).reshape(1,-1)
    prediction = model.predict(mw)
    print(prediction) 
    return prediction 
  
def main(): 
      
    st.title("Top 15 Prediction") 
    html_temp = ""
    
    
    urllib.request.urlretrieve('https://img.freepik.com/vector-premium/silueta-jugador-futbol-ilustracion-bola_62860-180.jpg',"JUGADOR.jpg")
    image = Image.open("JUGADOR.jpg").resize((300,400))
    st.image(image)
    st.markdown(html_temp, unsafe_allow_html = True) 
    goals_per_90_overall_input = st.number_input("Mean of goals per match") 
    assists_per_90_overall_input = st.number_input("Mean of assists per match") 
    goals_involved_per_90_overall_input = st.number_input("Mean of goales involved per match") 
    min_per_conceded_overall_input = st.number_input("Amount of minutes until a received goal against") 
    minutes_played_overall_input= st.number_input("Mean of amount of minutes played per match") 
    result =""
    #explainer_1,shap_values0=explainer(model)
    #shap_value = explainer_1.shap_values(np.array([goals_per_90_overall_input,assists_per_90_overall_input,goals_involved_per_90_overall_input,goals_involved_per_90_overall_input,min_per_conceded_overall_input]).reshape(1,-1))
    exp = explainer.explain_instance(np.array([goals_per_90_overall_input,assists_per_90_overall_input,goals_involved_per_90_overall_input,goals_involved_per_90_overall_input,min_per_conceded_overall_input]), model.predict_proba, num_features=6)

    
                                       
    
   


    
        

   

    if st.button("Predict"): 
        result =prediction(goals_per_90_overall_input,assists_per_90_overall_input,goals_involved_per_90_overall_input,goals_involved_per_90_overall_input,min_per_conceded_overall_input)
        
     
    
        if result==1:
            result='top 15 in his position'
        else:
            result= 'not top 15 in his position'
            
        st.success('The player is {}'.format(result)) 
    
    
        st.subheader('Analizando la prediccion:')
        
        #st_shap(shap.force_plot(explainer_1.expected_value, shap_value, np.array([goals_per_90_overall_input,assists_per_90_overall_input,goals_involved_per_90_overall_input,goals_involved_per_90_overall_input,min_per_conceded_overall_input]).reshape(1,-1)))
        components.html(exp.as_html(show_table=True), height=800)
                                       



if __name__=='__main__': 
    main() 
    


# In[2]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




