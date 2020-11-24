#!/usr/bin/env python
# coding: utf-8

# # UNDP: NajStudent

# ## *Pre-processing*

# In[1]:


#Importujem potrebne .py biblioteke
import pandas as pd
import numpy as np
import glob
import os
import re
import pandas.util.testing as tm
import statistics
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import plotly.graph_objects as go
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import iqr
from sklearn.linear_model import LinearRegression
from scipy import stats

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Učitavam .csv fajl
# pd.set_option('display.max_rows', 1655)
stipendije = pd.read_csv('C:\\Users\\isidora.gataric\\Desktop\\UNDP\\2. Podaci\\2. Avgust_2020\\FINAL\\najstudent_final_v2.csv', delimiter=';', encoding='utf-8')
stipendije


# In[3]:


# Lista varijabli koje imamo u ovom setu podataka
stipendije.columns.to_list()


# In[4]:


# Bazični opis data seta
stipendije.describe()


# In[5]:


# Proveravamo da li ima duplikata
stipendije.shape[0] != stipendije.drop_duplicates(['ID']).shape[0]


# In[6]:


# Provera postojanja null vrednosti 
stipendije.info()


# In[7]:


# Da proverimo da li imamo NaN/Null vrednosti
stipendije.isnull().values.any()


# In[8]:


# Pretvaram AverTimePage u ono što mi treba
stipendije['AverTimePageSeconds']=3600*pd.DatetimeIndex(stipendije['AverTimePage']).hour+60*pd.DatetimeIndex(stipendije['AverTimePage']).minute+pd.DatetimeIndex(stipendije['AverTimePage']).second
stipendije


# In[9]:


# Izbacujemo slučajeve gde je unique page views i prosečno vreme provdeno na stranici 0
stipendije.drop(stipendije.loc[stipendije['UniquePageviews']==0].index, inplace=True)
stipendije.drop(stipendije.loc[stipendije['AverTimePageSeconds']==0].index, inplace=True)
stipendije


# #### *Specijalni slučaj: Srbija*

# In[10]:


# Da vidim samo šta se dešava kod Srbije
srbija = stipendije.loc[stipendije['Zemlja'].isin(['Srbija'])]
srbija


# In[11]:


# Opisivanje podseta
srbija.describe()


# In[12]:


# Distribucija (iako to već sve vidimo prema BoxPlot (odnosno Mean/SD iz tabele iznad)) -- apsolutne vrednosti na y osi
sns.set(rc={'figure.figsize':(11.7,8.27)})
fig = sns.distplot(srbija['Year'], color="darkblue", label="Year", kde=False)
fig.set(xlabel='Godina', ylabel='Broj oglasa za stipendije (Srbija)')


# #### *EDA: Godine*

# In[13]:


# Gledamo da li postoje autlajeri za "Godina" (iako već vidim po MIN i MAX da nema, ali da budem sigurna) -- BoxPlot (nema potrebe za nečim drugim)
sns.boxplot(x=stipendije['Year'], color="darkblue")


# In[14]:


# Kako se vrednosti (broj stipendija) distribuira kroz godine (Plotly)
fig = px.histogram(stipendije, x="Year", nbins=20, color_discrete_sequence=['#518EBD'])
fig.update_layout(
    xaxis_title_text='Godina', # xaxis label
    yaxis_title_text='Broj oglasa', # yaxis label
)
fig.show()


# In[15]:


# Da proverimo kako stojimo sa brojem vrednosti za koju godinu
stipendije.groupby('Year').count()


# #### *EDA: Prosečno vreme provedeno na stranici*

# In[16]:


# Gledamo da li postoje autlajeri za "AverTimePage" (iako već vidim po MIN i MAX da nema, ali da budem sigurna) -- BoxPlot (nema potrebe za nečim drugim)
sns.boxplot(x=stipendije['AverTimePageSeconds'], color="darkblue")


# In[17]:


# Distribucija (iako to već sve vidimo prema BoxPlot (odnosno Mean/SD iz tabele iznad)) -- apsolutne vrednosti na y osi
sns.distplot(stipendije['AverTimePageSeconds'], color="darkblue", label="Year", kde=False)


# In[18]:


# Izbacujemo ekstremne vrednosti (autlajere)
for percentile in [0.95,0.96,0.97,0.98,0.99]:
    print(np.quantile(stipendije['AverTimePageSeconds'],percentile))


# In[19]:


# Uzeću kao granicu ovaj 3%, dakle 180.37
stipendije = stipendije.drop(stipendije[stipendije.AverTimePageSeconds > 180.37].index)
stipendije


# In[20]:


# Opis seta podataka
stipendije.describe()


# In[21]:


# Distribucija posle izbacivanja (Plotly)
fig = px.histogram(stipendije, x="AverTimePageSeconds", nbins=50, color_discrete_sequence=['#518EBD'])
fig.update_layout(
    xaxis_title_text='Prosečno vreme provedeno na stranici (sekunde)', # xaxis label
    yaxis_title_text='Broj oglasa', # yaxis label
)
fig.show()


# #### *EDA: Jedinstveni pregledi oglasa za stipendije*

# In[22]:


# Gledamo da li postoje autlajeri za "AverTimePage" (iako već vidim po MIN i MAX da nema, ali da budem sigurna) -- BoxPlot (nema potrebe za nečim drugim)
sns.boxplot(x=stipendije['UniquePageviews'], color="darkblue")


# In[23]:


# Distribucija (iako to već sve vidimo prema BoxPlot (odnosno Mean/SD iz tabele iznad)) -- apsolutne vrednosti na y osi
sns.distplot(stipendije['UniquePageviews'], color="darkblue", label="Year", kde=False)


# In[24]:


# Izbacujemo ekstremne vrednosti (autlajere)
for percentile in [0.95,0.96,0.97,0.98,0.99]:
    print(np.quantile(stipendije['UniquePageviews'],percentile))


# In[25]:


# Uzeću kao granicu ovaj 3%, dakle 180.37
stipendije = stipendije.drop(stipendije[stipendije.UniquePageviews > 1511.79].index)
stipendije


# In[26]:


# Distribucija posle izbacivanja (Plotly)
fig = px.histogram(stipendije, x="UniquePageviews", nbins=50, color_discrete_sequence=['#518EBD'])
fig.update_layout(
    xaxis_title_text='Jedinstveni pregledi stranica', # xaxis label
    yaxis_title_text='Broj oglasa', # yaxis label
)
fig.show()


# #### *EDA: Država*

# In[27]:


# Frekvencije, da vidimo iz kojih zemalja su stipendije
# stipendije['Zemlja'].unique().tolist()
stipendije['Zemlja'].value_counts()[:25]


# In[28]:


# Vizelizacija samo prvih 20
stipendije['Zemlja'].value_counts().head(20).plot(kind='barh', figsize=(20,10))


# In[29]:


# Izbacujemo sve slučajeve gde imamo "Srbija", "Online" i "EU", jer nam one ne trebaju (nisu relevantne za istraživački problem)
stipendije2 = stipendije[(stipendije.Zemlja != 'Srbija') & (stipendije.Zemlja != 'Online') & (stipendije.Zemlja != 'EU')]
stipendije2


# In[30]:


# Selektovanje top 10 za vizelni prikaz
stop10 = stipendije2[stipendije2["Zemlja"].isin(["UK","Nemačka", "Italija", "SAD", "Francuska", "Australija", "Austrija", "Španija", "Mađarska", "Poljska", "Rumunija"])]
# stop10


# In[31]:


# Vizuelni prikaz država (Plotly)
fig = px.histogram(stop10, y="Zemlja", nbins=20, orientation='h', color_discrete_sequence=['#518EBD'])
fig.update_layout(
    xaxis_title_text='Broj oglasa', # xaxis label
    yaxis_title_text='Država', # yaxis label
)
fig.show()


# #### *EDA: Kontinent*

# In[32]:


# Frekvencije, da vidimo iz kojih kontinenata su stipendije
# stipendije['Zemlja'].unique().tolist()
stipendije2['Kontinent'].value_counts()[:10]


# In[33]:


# Vizelizacija samo prvih 20
stipendije2['Kontinent'].value_counts().head(10).plot(kind='barh', figsize=(20,10))


# #### *EDA: Disciplina*

# In[34]:


# Lista svih vrednosti koje imamo u varijabli "Field"
# stipendije2['Disciplina'].unique().tolist()
stipendije2['Disciplina'].value_counts()[:25]


# In[35]:


# Vizelizacija samo prvih 20
stipendije2['Disciplina'].value_counts().head(20).plot(kind='barh', figsize=(20,10))


# In[36]:


# Selektovanje top 10 za vizelni prikaz
dtop10 = stipendije2[stipendije2["Disciplina"].isin(["Ekonomija, bankarstvo i finansije","Biološke nauke", "Informacione tehnologije", "Pravo", "Matematika", "Filologija", "Političke nauke", "Fizika", "Zaštita životne sredine (ekologija)", "Menadžment", "Medicina"])]
# dtop10


# In[37]:


# Plotly
fig = px.histogram(dtop10, y="Disciplina", nbins=20, orientation='h', color_discrete_sequence=['#518EBD'])
fig.update_layout(
    xaxis_title_text='Broj oglasa', # xaxis label
    yaxis_title_text='Disciplina', # yaxis label
)
fig.show()


# #### *EDA: Oblast studija*

# In[38]:


# Lista svih vrednosti koje imamo u varijabli "Field"
# stipendije2['OblastStudija'].unique().tolist()
stipendije2['OblastStudija'].value_counts()[:10]


# In[39]:


# Vizelni prikaz broja stipendija po oblastima studija
fig = px.histogram(stipendije2, y="OblastStudija", nbins=20, orientation='h', color_discrete_sequence=['#518EBD'])
fig.update_layout(
    xaxis_title_text='Broj oglasa', # xaxis label
    yaxis_title_text='Oblast studija', # yaxis label
)
fig.show()


# #### *EDA (ukrštanje): Oblast studija po godinama*

# In[40]:


# Prvo ćemo da vidimo iz kojih oblasti najčešće su te stipendije
stipendije2['OblastStudija'].value_counts()


# In[41]:


# Grupišemo po godinama
pd.get_dummies(stipendije2, columns=['Year']).groupby('OblastStudija').sum()


# In[42]:


# Selektujemo prva tri da vidimo šta se dešava po godinama
ns_stip_prvi = stipendije2.loc[stipendije2['OblastStudija'].isin(['Društvene nauke'])]
ns_stip_drugi = stipendije2.loc[stipendije2['OblastStudija'].isin(['Prirodne nauke'])]
ns_stip_treci = stipendije2.loc[stipendije2['OblastStudija'].isin(['Tehničke nauke'])]
ns_stip_cetiri = stipendije2.loc[stipendije2['OblastStudija'].isin(['Humanističke nauke'])]
ns_stip_pet = stipendije2.loc[stipendije2['OblastStudija'].isin(['Umetnost'])]
ns_stip_sest = stipendije2.loc[stipendije2['OblastStudija'].isin(['Medicinske nauke'])]
ns_stip_sedam = stipendije2.loc[stipendije2['OblastStudija'].isin(['Biotehničke nauke'])]
ns_stip_osam = stipendije2.loc[stipendije2['OblastStudija'].isin(['Sport i fizička kultura'])]


# In[43]:


# Prikaz po godinama -- Društvene nauke
sns.set(rc={'figure.figsize':(11.7,8.27)})
fig = sns.distplot(ns_stip_prvi['Year'], color="darkblue", label="Year", kde=False)
fig.set(xlabel='Godina', ylabel='Broj ponuđenih stipendija')


# In[44]:


# Prebrajanje slučajeva
ns_stip_prvi['Year'].value_counts()


# In[45]:


# Prikaz po godinama -- Prirodne nauke
sns.set(rc={'figure.figsize':(11.7,8.27)})
fig = sns.distplot(ns_stip_drugi['Year'], color="darkblue", label="Year", kde=False)
fig.set(xlabel='Godina', ylabel='Broj ponuđenih stipendija')


# In[46]:


# Prebrajanje slučajeva
ns_stip_drugi['Year'].value_counts()


# In[47]:


# Prikaz po godinama -- Tehničke nauke
sns.set(rc={'figure.figsize':(11.7,8.27)})
fig = sns.distplot(ns_stip_treci['Year'], color="darkblue", label="Year", kde=False)
fig.set(xlabel='Godina', ylabel='Broj ponuđenih stipendija')


# In[48]:


# Prebrajanje slučajeva
ns_stip_treci['Year'].value_counts()


# In[49]:


# Prikaz po godinama -- Humanističke nauke
sns.set(rc={'figure.figsize':(11.7,8.27)})
fig = sns.distplot(ns_stip_cetiri['Year'], color="darkblue", label="Year", kde=False)
fig.set(xlabel='Godina', ylabel='Broj ponuđenih stipendija')


# In[50]:


# Prebrajanje slučajeva
ns_stip_cetiri['Year'].value_counts()


# In[51]:


# Prikaz po godinama -- Umetnost
sns.set(rc={'figure.figsize':(11.7,8.27)})
fig = sns.distplot(ns_stip_pet['Year'], color="darkblue", label="Year", kde=False)
fig.set(xlabel='Godina', ylabel='Broj ponuđenih stipendija')


# In[52]:


# Prebrajanje slučajeva
ns_stip_pet['Year'].value_counts()


# In[53]:


# Prikaz po godinama -- Medicina
sns.set(rc={'figure.figsize':(11.7,8.27)})
fig = sns.distplot(ns_stip_sest['Year'], color="darkblue", label="Year", kde=False)
fig.set(xlabel='Godina', ylabel='Broj ponuđenih stipendija')


# In[54]:


# Prebrajanje slučajeva
ns_stip_sest['Year'].value_counts()


# In[55]:


# Prikaz po godinama -- Biotehničke nauke
sns.set(rc={'figure.figsize':(11.7,8.27)})
fig = sns.distplot(ns_stip_sedam['Year'], color="darkblue", label="Year", kde=False)
fig.set(xlabel='Godina', ylabel='Broj ponuđenih stipendija')


# In[56]:


# Prebrajanje slučajeva
ns_stip_sedam['Year'].value_counts()


# In[57]:


# Prikaz po godinama -- Sport i fizička kultura
sns.set(rc={'figure.figsize':(11.7,8.27)})
fig = sns.distplot(ns_stip_osam['Year'], color="darkblue", label="Year", kde=False)
fig.set(xlabel='Godina', ylabel='Broj ponuđenih stipendija')


# In[58]:


# Prebrajanje slučajeva
ns_stip_osam['Year'].value_counts()


# #### *EDA (ukrštanje): Jedinstveni pregledi oglasa po godinama*

# In[59]:


# Učitavamo sumirane podatke, da možemo da prikažemo lepo po godinama
stud = pd.read_csv('C:\\Users\\isidora.gataric\\Desktop\\UNDP\\2. Podaci\\2. Avgust_2020\\FINAL\\najstudent_god_sume.csv', delimiter=';', encoding='utf-8')
# stud


# In[60]:


# Proj pregleda oglasa (unique) po godinama (od 2013 do 2020)
fig = sns.lineplot(x="Godina", y="UniquePageviews", ci=None, data=stud)
fig.set(xlabel='Godina', ylabel='Broj jedinstvenih pregleda stranice')


# #### *EDA (ukrštanje): Prosečno vreme provedeno na stranici po godinama*

# In[61]:


# Proj pregleda po godinama (od 2013 do 2020)
fig = sns.lineplot(x="Godina", y="AverTimePageSeconds", ci=None, data=stud)
fig.set(xlabel='Godina', ylabel='Prosečno vreme provedeno na stranici (sekunde)')


# #### *EDA (ukrštanje): Jedinstveni pregledi stranice po državama*

# In[62]:


# Grafički prikaz (definitivno, bolje napraviti nadređene kategorije)
plt.figure(figsize=(20,10))
sns.boxplot(
    data=stipendije2,
    x='UniquePageviews',
    y='Zemlja',
#    hue='Year',
    color='red')


# #### *EDA (ukrštanje): Jedinstveni pregledi stranice po kontinentima*

# In[63]:


# Grafički prikaz (definitivno, bolje napraviti nadređene kategorije)
plt.figure(figsize=(20,10))
sns.boxplot(
    data=stipendije2,
    x='UniquePageviews',
    y='Kontinent',
#    hue='Year',
    color='red')


# #### *EDA (ukrštanje): Prosečno vreme provedeno na stranici po državama*

# In[64]:


# Grafički prikaz (definitivno, bolje napraviti nadređene kategorije)
plt.figure(figsize=(20,10))
sns.boxplot(
    data=stipendije2,
    x='AverTimePageSeconds',
    y='Zemlja',
#    hue='Year',
    color='red')


# #### *EDA (ukrštanje): Prosečno vreme provedeno na stranici po kontinentima*

# In[65]:


# Grafički prikaz (definitivno, bolje napraviti nadređene kategorije)
plt.figure(figsize=(20,10))
sns.boxplot(
    data=stipendije2,
    x='AverTimePageSeconds',
    y='Kontinent',
#    hue='Year',
    color='red')


# #### *EDA (ukrštanje): Prosečno vreme provedeno na stranici po oglasti studija*

# In[66]:


# Učitavamo sumirane podatke, da možemo da prikažemo lepo po godinama
studi = pd.read_csv('C:\\Users\\isidora.gataric\\Desktop\\UNDP\\2. Podaci\\2. Avgust_2020\\FINAL\\najstudent_oblast_sume.csv', delimiter=';', encoding='utf-8')
# stud


# In[67]:


# Proj pregleda po godinama (od 2013 do 2020)
sns.set(rc={'figure.figsize':(15,9)})
fig = sns.lineplot(x="OblastStudija", y="AverTimePageSeconds", ci=None, data=studi)
fig.set(xlabel='Oblast studija', ylabel='Prosečno vreme provedeno na stranici (sekunde)')


# In[68]:


# Plotly
fig = px.bar(studi, x="AverTimePageSeconds", y="OblastStudija", orientation='h', color_discrete_sequence=['#518EBD'])
fig.update_layout(
    xaxis_title_text='Prosečno vreme provedeno na stranici (sekunde)', # xaxis label
    yaxis_title_text='Oblast studija', # yaxis label
)
fig.show()


# #### *EDA (ukrštanje): Jedinstveni pregled stranica po oglasti studija*

# In[69]:


# Plotly
fig = px.bar(studi, x="UniquePageviews", y="OblastStudija", orientation='h', color_discrete_sequence=['#518EBD'])
fig.update_layout(
    xaxis_title_text='Jedinstveni pregledi stranica', # xaxis label
    yaxis_title_text='Oblast studija', # yaxis label
)
fig.show()


# #### *EDA (ukrštanje): Države po godinama*

# In[70]:


# Grupišemo po godinama
pd.get_dummies(stipendije2, columns=['Year']).groupby('Zemlja').sum()


# #### *EDA (ukrštanje): Discipline po godinama (top 3)*

# In[71]:


# Selektujemo prva tri da vidimo šta se dešava po godinama
prva = stipendije2.loc[stipendije2['Disciplina'].isin(['Ekonomija, bankarstvo i finansije'])]
druga = stipendije2.loc[stipendije2['Disciplina'].isin(['Biološke nauke'])]
treca = stipendije2.loc[stipendije2['Disciplina'].isin(['Informacione tehnologije'])]


# In[72]:


# Prikaz po godinama -- ekonomija, bankarstvo
sns.set(rc={'figure.figsize':(11.7,8.27)})
fig = sns.distplot(prva['Year'], color="darkblue", label="Year", kde=False)
fig.set(xlabel='Godina', ylabel='Broj ponuđenih stipendija')


# In[73]:


# Prebrajanje slučajeva po godinama
prva['Year'].value_counts()


# In[74]:


# Prebrajanje slučajeva po državama
prva['Zemlja'].value_counts()


# In[75]:


# Prikaz po godinama -- biološke nauke
sns.set(rc={'figure.figsize':(11.7,8.27)})
fig = sns.distplot(druga['Year'], color="blue", label="Year", kde=False)
fig.set(xlabel='Godina', ylabel='Broj ponuđenih stipendija')


# In[76]:


# Prebrajanje slučajeva po godinama
druga['Year'].value_counts()


# In[77]:


# Prebrajanje slučajeva po državama
druga['Zemlja'].value_counts()


# In[78]:


# Prikaz po godinama -- IT
sns.set(rc={'figure.figsize':(11.7,8.27)})
fig = sns.distplot(treca['Year'], color="lightblue", label="Year", kde=False)
fig.set(xlabel='Godina', ylabel='Broj ponuđenih stipendija')


# In[79]:


# Prebrajanje slučajeva po godinama
treca['Year'].value_counts()


# In[80]:


# Prebrajanje slučajeva po državama
treca['Zemlja'].value_counts()


# In[81]:


# Prikazujemo grafički objedinjeno
fig = sns.distplot(prva['Year'], color="darkblue", label="Year", kde=False)
fig = sns.distplot(druga['Year'], color="blue", label="Year", kde=False)
fig = sns.distplot(treca['Year'], color="lightblue", label="Year", kde=False)
fig.set(xlabel='Godina', ylabel='Broj ponuđenih stipendija')


# #### *EDA (ukrštanje): Države po godinama (top 3)*

# In[82]:


# Selektujemo prva tri da vidimo šta se dešava po godinama
prva1 = stipendije2.loc[stipendije2['Zemlja'].isin(['UK'])]
druga2 = stipendije2.loc[stipendije2['Zemlja'].isin(['Nemačka'])]
treca3 = stipendije2.loc[stipendije2['Zemlja'].isin(['Italija'])]


# In[83]:


# Prikaz po godinama -- UK
sns.set(rc={'figure.figsize':(11.7,8.27)})
fig = sns.distplot(prva1['Year'], color="darkblue", label="Year", kde=False)
fig.set(xlabel='Godina', ylabel='Broj ponuđenih stipendija')


# In[84]:


# Prebrajanje slučajeva po godinama
prva1['Year'].value_counts()


# In[85]:


# Prikaz po godinama -- Nemačka
sns.set(rc={'figure.figsize':(11.7,8.27)})
fig = sns.distplot(druga2['Year'], color="blue", label="Year", kde=False)
fig.set(xlabel='Godina', ylabel='Broj ponuđenih stipendija')


# In[86]:


# Prebrajanje slučajeva po godinama
druga2['Year'].value_counts()


# In[87]:


# Prikaz po godinama -- Italija
sns.set(rc={'figure.figsize':(11.7,8.27)})
fig = sns.distplot(treca3['Year'], color="lightblue", label="Year", kde=False)
fig.set(xlabel='Godina', ylabel='Broj ponuđenih stipendija')


# In[88]:


# Prebrajanje slučajeva po godinama
treca3['Year'].value_counts()


# #### *EDA: Korelacija (pre Regresione analize - ML)*

# In[89]:


# Pirsonova korelacija (mada ovo nije najbolji izbor, probaću i sa )
korelacija = stipendije2[['Pageviews','UniquePageviews','AverTimePageSeconds']]
korelacija.corr(method='pearson', min_periods=1)


# In[90]:


# Probam i Kendal-tau, takođe previsoka
korelacija.corr(method='kendall', min_periods=1)


# **Note:** Pageviews i UniquePageviews korelacija je 0.84 (Pirson), 0.98 (Kendal Tau) -- obe veoma visoke. Dakle, idemo dalje samo sa Unique Pageviews. Što se tiče "AverPageTime" i "UniquePageViews" tu možemo regresiju dalje raditi (iako je korelacija veoma, veoma niska, ne očekujem ništa), ali čisto da vidimo da li beležimo neki patern.

# ## *ML -- Regresiona analiza*

# ## *Linearna regresija po oblastima studija/godinama*

# #### *IQR po natkategorijama*

# In[91]:


# Definisanje fukncije za IQR
def ciscenje (datacolumn):
    sorted(datacolumn)
    Q1,Q3 = np.percentile(datacolumn, [25,75])
    IQR = Q3 - Q1
    locmin = Q1 - (1.5 * IQR)
    locmax = Q3 + (1.5 * IQR)
    return locmin, locmax


# In[92]:


# Čišćenje lokalnih min/max vrednosti
kategorije=stipendije2.copy()
for y in list (dict.fromkeys(kategorije['OblastStudija'].values)):
    for m in ['UniquePageviews', 'AverTimePageSeconds']:
        d=kategorije[(kategorije['OblastStudija']==y)][m]
        if d.size>0:
            locmin, locmax=ciscenje(d)
            kategorije.drop(index=kategorije.index[(kategorije['OblastStudija']==y)&(kategorije[m]<locmin)], inplace=True)
            kategorije.drop(index=kategorije.index[(kategorije['OblastStudija']==y)&(kategorije[m]>locmax)], inplace=True)  


# In[93]:


# Da vidimo koli nam je set podataka sada
kategorije.shape


# ### *Linearna regresija (standardna) po oblastima studija*

# #### *LR (standardna) -- Prirodne nauke*

# In[94]:


# Regresija (skater)
df1= kategorije.loc[kategorije['OblastStudija'].isin(['Prirodne nauke'])]
X = df1.iloc[:, 3].values.reshape(-1, 1)
Y = df1.iloc[:, 10].values.reshape(-1, 1)
linear_regressor = LinearRegression()
r = linear_regressor.fit(X, Y)
Y_pred = linear_regressor.predict(X)

plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()


# In[95]:


# Regresioni skor
r.score(X,Y)


# In[96]:


# Regresioni koeficijent
r.coef_


# #### *LR (standardna) -- Društvene nauke*

# In[97]:


# Regresija (skater)
df2= kategorije.loc[kategorije['OblastStudija'].isin(['Društvene nauke'])]
X = df2.iloc[:, 3].values.reshape(-1, 1)
Y = df2.iloc[:, 10].values.reshape(-1, 1)
linear_regressor = LinearRegression()
r = linear_regressor.fit(X, Y)
Y_pred = linear_regressor.predict(X)

plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()


# In[98]:


# Regresioni skor
r.score(X,Y)


# In[99]:


# Regresioni koeficijent
r.coef_


# #### *LR (standardna) -- Umetnost*

# In[100]:


# Regresija (skater)
df3= kategorije.loc[kategorije['OblastStudija'].isin(['Umetnost'])]
X = df3.iloc[:, 3].values.reshape(-1, 1)
Y = df3.iloc[:, 10].values.reshape(-1, 1)
linear_regressor = LinearRegression()
r = linear_regressor.fit(X, Y)
Y_pred = linear_regressor.predict(X)

plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()


# In[101]:


# Regresioni skor
r.score(X,Y)


# In[102]:


# Regresioni koeficijent
r.coef_


# #### *LR (standardna) -- Tehničke nauke*

# In[103]:


# Regresija (skater)
df4= kategorije.loc[kategorije['OblastStudija'].isin(['Tehničke nauke'])]
X = df4.iloc[:, 3].values.reshape(-1, 1)
Y = df4.iloc[:, 10].values.reshape(-1, 1)
linear_regressor = LinearRegression()
r = linear_regressor.fit(X, Y)
Y_pred = linear_regressor.predict(X)

plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()


# In[104]:


# Regresioni skor
r.score(X,Y)


# In[105]:


# Regresioni koeficijent
r.coef_


# #### *LR (standardna) -- Medicinske nauke*

# In[106]:


# Regresija (skater)
df5= kategorije.loc[kategorije['OblastStudija'].isin(['Medicinske nauke'])]
X = df5.iloc[:, 3].values.reshape(-1, 1)
Y = df5.iloc[:, 10].values.reshape(-1, 1)
linear_regressor = LinearRegression()
r = linear_regressor.fit(X, Y)
Y_pred = linear_regressor.predict(X)

plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()


# In[107]:


# Regresioni skor
r.score(X,Y)


# In[108]:


# Regresioni koeficijent
r.coef_


# #### *LR (standardna) -- Biotehničke nauke*

# In[109]:


# Regresija (skater)
df6= kategorije.loc[kategorije['OblastStudija'].isin(['Biotehničke nauke'])]
X = df6.iloc[:, 3].values.reshape(-1, 1)
Y = df6.iloc[:, 10].values.reshape(-1, 1)
linear_regressor = LinearRegression()
r = linear_regressor.fit(X, Y)
Y_pred = linear_regressor.predict(X)

plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()


# In[110]:


# Regresioni skor
r.score(X,Y)


# In[111]:


# Regresioni koeficijent
r.coef_


# #### *LR (standardna) -- Humanističke nauke*

# In[112]:


# Regresija (skater)
df7= kategorije.loc[kategorije['OblastStudija'].isin(['Humanističke nauke'])]
X = df7.iloc[:, 3].values.reshape(-1, 1)
Y = df7.iloc[:, 10].values.reshape(-1, 1)
linear_regressor = LinearRegression()
r = linear_regressor.fit(X, Y)
Y_pred = linear_regressor.predict(X)

plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()


# In[113]:


# Regresioni skor
r.score(X,Y)


# In[114]:


# Regresioni koeficijent
r.coef_


# #### *LR (standardna) -- Sport i fizička kultura*

# In[115]:


# Regresija (skater)
df8= kategorije.loc[kategorije['OblastStudija'].isin(['Sport i fizička kultura'])]
X = df8.iloc[:, 3].values.reshape(-1, 1)
Y = df8.iloc[:, 10].values.reshape(-1, 1)
linear_regressor = LinearRegression()
r = linear_regressor.fit(X, Y)
Y_pred = linear_regressor.predict(X)

plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()


# In[116]:


# Regresioni skor
r.score(X,Y)


# In[117]:


# Regresioni koeficijent
r.coef_


# ### *Linearna regresija (modifikovana -- median/round) po oblastima studija*

# #### *Transformacija postojećih varijable u modifikovane (median/round)*

# In[118]:


# Prvo treba napraviti te varijable u setu koji vec imamo, krecemo od seta kategorije (round)
kategorije['UniquePageviewsRounded']=kategorije['UniquePageviews'].round(-2)
kategorije
#'UniquePageviews', 'AverTimePageSeconds'


# In[119]:


# Median za BrojKonkurisanja
df = pd.DataFrame(columns=['OblastStudija','UniquePageviewsRounded','size','AverTimePageSecondsMedian'])


# In[120]:


# Upisujemo vrednosti u dve nove varijable Round/Median
for n in list(dict.fromkeys(kategorije['OblastStudija'].values)):
    for r in list(dict.fromkeys(kategorije['UniquePageviewsRounded'].values)):
        extr_app=[]
        extr_app.append(n)
        extr_app.append(r)
        d=kategorije[(kategorije['OblastStudija']==n)& (kategorije['UniquePageviewsRounded']==r)]['AverTimePageSeconds']
#       print(d.size)
        if d.size>8:
            extr_app.append(d.size)
            extr_app.append(statistics.median(d.tolist()))
#            print(extr_app)
            df=df.append(pd.Series(extr_app,index=df.columns), ignore_index=True)
df


# #### *LR (modifikovana) -- Prirodne nauke*

# In[121]:


# Regresija (skater)
df1= df.loc[df['OblastStudija'].isin(['Prirodne nauke'])]
X = df1.iloc[:, 1].values.reshape(-1, 1)
Y = df1.iloc[:, 3].values.reshape(-1, 1)
linear_regressor = LinearRegression()
r = linear_regressor.fit(X, Y)
Y_pred = linear_regressor.predict(X)

plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()


# In[122]:


# Regresioni skor
r.score(X,Y)


# In[123]:


# Regresioni koeficijent
r.coef_


# #### *LR (modifikovana) -- Društvene nauke*

# In[124]:


# Regresija (skater)
df2= df.loc[df['OblastStudija'].isin(['Društvene nauke'])]
X = df2.iloc[:, 1].values.reshape(-1, 1)
Y = df2.iloc[:, 3].values.reshape(-1, 1)
linear_regressor = LinearRegression()
r = linear_regressor.fit(X, Y)
Y_pred = linear_regressor.predict(X)

plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()


# In[125]:


# Regresioni skor
r.score(X,Y)


# In[126]:


# Regresioni koeficijent
r.coef_


# #### *LR (modifikovana) -- Umetnost*

# In[127]:


# Regresija (skater)
df3= df.loc[df['OblastStudija'].isin(['Umetnost'])]
X = df3.iloc[:, 1].values.reshape(-1, 1)
Y = df3.iloc[:, 3].values.reshape(-1, 1)
linear_regressor = LinearRegression()
r = linear_regressor.fit(X, Y)
Y_pred = linear_regressor.predict(X)

plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()


# In[128]:


# Regresioni skor
r.score(X,Y)


# In[129]:


# Regresioni koeficijent
r.coef_


# #### *LR (modifikovana) -- Tehničke nauke*

# In[130]:


# Regresija (skater)
df4= df.loc[df['OblastStudija'].isin(['Tehničke nauke'])]
X = df4.iloc[:, 1].values.reshape(-1, 1)
Y = df4.iloc[:, 3].values.reshape(-1, 1)
linear_regressor = LinearRegression()
r = linear_regressor.fit(X, Y)
Y_pred = linear_regressor.predict(X)

plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()


# In[131]:


# Regresioni skor
r.score(X,Y)


# In[132]:


# Regresioni koeficijent
r.coef_


# #### *LR (modifikovana) -- Medicinske nauke*

# In[133]:


# Regresija (skater)
df5= df.loc[df['OblastStudija'].isin(['Medicinske nauke'])]
X = df5.iloc[:, 1].values.reshape(-1, 1)
Y = df5.iloc[:, 3].values.reshape(-1, 1)
linear_regressor = LinearRegression()
r = linear_regressor.fit(X, Y)
Y_pred = linear_regressor.predict(X)

plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()


# **Note**: nema materijala za regresiju.

# #### *LR (modifikovana) -- Biotehničke nauke*

# In[134]:


# Regresija (skater)
df6= df.loc[df['OblastStudija'].isin(['Biotehničke nauke'])]
X = df6.iloc[:, 1].values.reshape(-1, 1)
Y = df6.iloc[:, 3].values.reshape(-1, 1)
linear_regressor = LinearRegression()
r = linear_regressor.fit(X, Y)
Y_pred = linear_regressor.predict(X)

plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()


# **Note**: nema materijala za regresiju.

# #### *LR (modifikovana) -- Humanističke nauke*

# In[135]:


# Regresija (skater)
df7= df.loc[df['OblastStudija'].isin(['Humanističke nauke'])]
X = df7.iloc[:, 1].values.reshape(-1, 1)
Y = df7.iloc[:, 3].values.reshape(-1, 1)
linear_regressor = LinearRegression()
r = linear_regressor.fit(X, Y)
Y_pred = linear_regressor.predict(X)

plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()


# In[136]:


# Regresioni skor
r.score(X,Y)


# In[137]:


# Regresioni koeficijent
r.coef_


# #### *LR (modifikovana) -- Sport i fizička kultura*

# In[138]:


# Regresija (skater)
#df8= df.loc[df['OblastStudija'].isin(['Sport i fizička kultura'])]
#X = df8.iloc[:, 1].values.reshape(-1, 1)
#Y = df8.iloc[:, 3].values.reshape(-1, 1)
#linear_regressor = LinearRegression()
#r = linear_regressor.fit(X, Y)
#Y_pred = linear_regressor.predict(X)
#
#plt.scatter(X, Y)
#plt.plot(X, Y_pred, color='red')
#plt.show()


# **Note**: nema materijala za regresiju, postoji samo 1 data point, pa nam izbacuje error.

# #### *IQR po godinama*

# In[139]:


# Definisanje fukncije za IQR
def ciscenje (datacolumn):
    sorted(datacolumn)
    Q1,Q3 = np.percentile(datacolumn, [25,75])
    IQR = Q3 - Q1
    locmin = Q1 - (1.5 * IQR)
    locmax = Q3 + (1.5 * IQR)
    return locmin, locmax


# In[140]:


# Čišćenje lokalnih min/max vrednosti
kategorije1=stipendije2.copy()
for y in list (dict.fromkeys(kategorije1['Year'].values)):
    for m in ['UniquePageviews', 'AverTimePageSeconds']:
        d=kategorije1[(kategorije1['Year']==y)][m]
        if d.size>0:
            locmin, locmax=ciscenje(d)
            kategorije1.drop(index=kategorije1.index[(kategorije1['Year']==y)&(kategorije1[m]<locmin)], inplace=True)
            kategorije1.drop(index=kategorije1.index[(kategorije1['Year']==y)&(kategorije1[m]>locmax)], inplace=True) 


# In[141]:


# Gledamo kako nam izgleda sada data set
kategorije1.shape


# ### *Linearna regresija (standardna) po godinama*

# #### *LR (standardna) -- 2013*

# In[142]:


# Regresija (skater)
df1= kategorije1.loc[kategorije1['Year'].isin([2013])]
X = df1.iloc[:, 3].values.reshape(-1, 1)
Y = df1.iloc[:, 10].values.reshape(-1, 1)
linear_regressor = LinearRegression()
r = linear_regressor.fit(X, Y)
Y_pred = linear_regressor.predict(X)

plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()


# In[143]:


# Regresioni skor
r.score(X,Y)


# In[144]:


# Regresioni koeficijent
r.coef_


# #### *LR (standardna) -- 2014*

# In[145]:


# Regresija (skater)
df2= kategorije1.loc[kategorije1['Year'].isin([2014])]
X = df2.iloc[:, 3].values.reshape(-1, 1)
Y = df2.iloc[:, 10].values.reshape(-1, 1)
linear_regressor = LinearRegression()
r = linear_regressor.fit(X, Y)
Y_pred = linear_regressor.predict(X)

plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()


# In[146]:


# Regresioni skor
r.score(X,Y)


# In[147]:


# Regresioni koeficijent
r.coef_


# #### *LR (standardna) -- 2015*

# In[148]:


# Regresija (skater)
df3= kategorije1.loc[kategorije1['Year'].isin([2015])]
X = df3.iloc[:, 3].values.reshape(-1, 1)
Y = df3.iloc[:, 10].values.reshape(-1, 1)
linear_regressor = LinearRegression()
r = linear_regressor.fit(X, Y)
Y_pred = linear_regressor.predict(X)

plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()


# In[149]:


# Regresioni skor
r.score(X,Y)


# In[150]:


# Regresioni koeficijent
r.coef_


# #### *LR (standardna) -- 2016*

# In[151]:


# Regresija (skater)
df4= kategorije1.loc[kategorije1['Year'].isin([2016])]
X = df4.iloc[:, 3].values.reshape(-1, 1)
Y = df4.iloc[:, 10].values.reshape(-1, 1)
linear_regressor = LinearRegression()
r = linear_regressor.fit(X, Y)
Y_pred = linear_regressor.predict(X)

plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()


# In[152]:


# Regresioni skor
r.score(X,Y)


# In[153]:


# Regresioni koeficijent
r.coef_


# #### *LR (standardna) -- 2017*

# In[154]:


# Regresija (skater)
df5= kategorije1.loc[kategorije1['Year'].isin([2017])]
X = df5.iloc[:, 3].values.reshape(-1, 1)
Y = df5.iloc[:, 10].values.reshape(-1, 1)
linear_regressor = LinearRegression()
r = linear_regressor.fit(X, Y)
Y_pred = linear_regressor.predict(X)

plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()


# In[155]:


# Regresioni skor
r.score(X,Y)


# In[156]:


# Regresioni koeficijent
r.coef_


# #### *LR (standardna) -- 2018*

# In[157]:


# Regresija (skater)
df6= kategorije1.loc[kategorije1['Year'].isin([2018])]
X = df6.iloc[:, 3].values.reshape(-1, 1)
Y = df6.iloc[:, 10].values.reshape(-1, 1)
linear_regressor = LinearRegression()
r = linear_regressor.fit(X, Y)
Y_pred = linear_regressor.predict(X)

plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()


# In[158]:


# Regresioni skor
r.score(X,Y)


# In[159]:


# Regresioni koeficijent
r.coef_


# #### *LR (standardna) -- 2019*

# In[160]:


# Regresija (skater)
df7= kategorije1.loc[kategorije1['Year'].isin([2019])]
X = df7.iloc[:, 3].values.reshape(-1, 1)
Y = df7.iloc[:, 10].values.reshape(-1, 1)
linear_regressor = LinearRegression()
r = linear_regressor.fit(X, Y)
Y_pred = linear_regressor.predict(X)

plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()


# In[161]:


# Regresioni skor
r.score(X,Y)


# In[162]:


# Regresioni koeficijent
r.coef_


# #### *LR (standardna) -- 2020*

# In[163]:


# Regresija (skater)
df8= kategorije1.loc[kategorije1['Year'].isin([2020])]
X = df8.iloc[:, 3].values.reshape(-1, 1)
Y = df8.iloc[:, 10].values.reshape(-1, 1)
linear_regressor = LinearRegression()
r = linear_regressor.fit(X, Y)
Y_pred = linear_regressor.predict(X)

plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()


# In[164]:


# Regresioni skor
r.score(X,Y)


# In[165]:


# Regresioni koeficijent
r.coef_


# ### *Linearna regresija (modifikovana -- median/round) po godinama*

# #### *Transformacija postojećih varijable u modifikovane (median/round)*

# In[166]:


# Prvo treba napraviti te varijable u setu koji vec imamo, krecemo od seta kategorije (round)
kategorije1['UniquePageviewsRounded']=kategorije1['UniquePageviews'].round(-2)
kategorije1
#'UniquePageviews', 'AverTimePageSeconds'


# In[167]:


# Median za BrojKonkurisanja
df1 = pd.DataFrame(columns=['Year','UniquePageviewsRounded','size','AverTimePageSecondsMedian'])


# In[168]:


# Upisujemo vrednosti u dve nove varijable Round/Median
for n in list(dict.fromkeys(kategorije1['Year'].values)):
    for r in list(dict.fromkeys(kategorije1['UniquePageviewsRounded'].values)):
        extr_app=[]
        extr_app.append(n)
        extr_app.append(r)
        d=kategorije1[(kategorije1['Year']==n)& (kategorije1['UniquePageviewsRounded']==r)]['AverTimePageSeconds']
#       print(d.size)
        if d.size>8:
            extr_app.append(d.size)
            extr_app.append(statistics.median(d.tolist()))
#            print(extr_app)
            df1=df1.append(pd.Series(extr_app,index=df1.columns), ignore_index=True)
df1


# #### *LR (modifikovana) -- 2013*

# In[169]:


# Regresija (skater)
df2= df1.loc[df1['Year'].isin([2013])]
X = df2.iloc[:, 1].values.reshape(-1, 1)
Y = df2.iloc[:, 3].values.reshape(-1, 1)
linear_regressor = LinearRegression()
r = linear_regressor.fit(X, Y)
Y_pred = linear_regressor.predict(X)

plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()


# In[170]:


# Regresioni skor
r.score(X,Y)


# In[171]:


# Regresioni koeficijent
r.coef_


# #### *LR (modifikovana) -- 2014*

# In[172]:


# Regresija (skater)
df3= df1.loc[df1['Year'].isin([2014])]
X = df3.iloc[:, 1].values.reshape(-1, 1)
Y = df3.iloc[:, 3].values.reshape(-1, 1)
linear_regressor = LinearRegression()
r = linear_regressor.fit(X, Y)
Y_pred = linear_regressor.predict(X)

plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()


# In[173]:


# Regresioni skor
r.score(X,Y)


# In[174]:


# Regresioni koeficijent
r.coef_


# #### *LR (modifikovana) -- 2015*

# In[175]:


# Regresija (skater)
df4= df1.loc[df1['Year'].isin([2015])]
X = df4.iloc[:, 1].values.reshape(-1, 1)
Y = df4.iloc[:, 3].values.reshape(-1, 1)
linear_regressor = LinearRegression()
r = linear_regressor.fit(X, Y)
Y_pred = linear_regressor.predict(X)

plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()


# In[176]:


# Regresioni skor
r.score(X,Y)


# In[177]:


# Regresioni koeficijent
r.coef_


# #### *LR (modifikovana) -- 2016*

# In[178]:


# Regresija (skater)
df5= df1.loc[df1['Year'].isin([2016])]
X = df5.iloc[:, 1].values.reshape(-1, 1)
Y = df5.iloc[:, 3].values.reshape(-1, 1)
linear_regressor = LinearRegression()
r = linear_regressor.fit(X, Y)
Y_pred = linear_regressor.predict(X)

plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()


# In[179]:


# Regresioni skor
r.score(X,Y)


# In[180]:


# Regresioni koeficijent
r.coef_


# #### *LR (modifikovana) -- 2017*

# In[181]:


# Regresija (skater)
df6= df1.loc[df1['Year'].isin([2017])]
X = df6.iloc[:, 1].values.reshape(-1, 1)
Y = df6.iloc[:, 3].values.reshape(-1, 1)
linear_regressor = LinearRegression()
r = linear_regressor.fit(X, Y)
Y_pred = linear_regressor.predict(X)

plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()


# In[182]:


# Regresioni skor
r.score(X,Y)


# In[183]:


# Regresioni koeficijent
r.coef_


# #### *LR (modifikovana) -- 2018*

# In[184]:


# Regresija (skater)
df6= df1.loc[df1['Year'].isin([2018])]
X = df6.iloc[:, 1].values.reshape(-1, 1)
Y = df6.iloc[:, 3].values.reshape(-1, 1)
linear_regressor = LinearRegression()
r = linear_regressor.fit(X, Y)
Y_pred = linear_regressor.predict(X)

plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()


# In[185]:


# Regresioni skor
r.score(X,Y)


# In[186]:


# Regresioni koeficijent
r.coef_


# #### *LR (modifikovana) -- 2019*

# In[187]:


# Regresija (skater)
df8= df1.loc[df1['Year'].isin([2019])]
X = df8.iloc[:, 1].values.reshape(-1, 1)
Y = df8.iloc[:, 3].values.reshape(-1, 1)
linear_regressor = LinearRegression()
r = linear_regressor.fit(X, Y)
Y_pred = linear_regressor.predict(X)

plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()


# In[188]:


# Regresioni skor
r.score(X,Y)


# In[189]:


# Regresioni koeficijent
r.coef_


# #### *LR (modifikovana) -- 2020*

# In[190]:


# Regresija (skater)
df9= df1.loc[df1['Year'].isin([2020])]
X = df8.iloc[:, 1].values.reshape(-1, 1)
Y = df8.iloc[:, 3].values.reshape(-1, 1)
linear_regressor = LinearRegression()
r = linear_regressor.fit(X, Y)
Y_pred = linear_regressor.predict(X)

plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()


# In[191]:


# Regresioni skor
r.score(X,Y)


# In[192]:


# Regresioni koeficijent
r.coef_

