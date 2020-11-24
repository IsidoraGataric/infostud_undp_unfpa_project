#!/usr/bin/env python
# coding: utf-8

# # UNDP: Poslovi

# ## *Pre-processing*

# In[1]:


#Importujem potrebne .py biblioteke
import pandas as pd
import numpy as np
import glob
import os
import re
import nltk
import pandas.util.testing as tm
import statistics
import seaborn as sns
import plotly.express as px
from nltk.corpus import PlaintextCorpusReader
import plotly.graph_objects as go
from nltk.util import ngrams
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.text import Text
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import iqr
from sklearn.linear_model import LinearRegression
from scipy import stats

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Učitavanje seta podataka (.csv fajl)
poslovi = pd.read_csv('C:\\Users\\isidora.gataric\\Desktop\\UNDP\\2. Podaci\\2. Avgust_2020\\FINAL\\poslovi_final_v6.csv', delimiter=';', encoding='utf-8')
poslovi


# In[3]:


# Lista varijabli koje imamo u setu podataka
poslovi.columns.to_list()


# In[4]:


# Pravimo posebnu varijablu "Mesec" i "Dan", ekstrahujemo informaciju iz datuma
poslovi['Mesec_proba'] = pd.DatetimeIndex(poslovi['DatumObjave']).month
poslovi['Dan_proba'] = pd.DatetimeIndex(poslovi['DatumObjave']).day
poslovi.head()


# In[5]:


# Proveravamo da li ima duplikata
poslovi.shape[0] != poslovi.drop_duplicates(['ID']).shape[0]
# poslovi.drop_duplicates(['ID'])


# In[6]:


# Čistimo set podataka od suvišnih varijabli, ostavljamo samo ono što je od značaja trenutno
poslovi = poslovi[['ID', 'BrojKonkurisanja', 'BrojPregledaOglasa', 'NacinKonkurisanja', 'PrimarnaKategorija', 'StandardizovanaPozicija', 'Drzava', 'ImaInostranstvoGrad', 'Regioni', 'ImaInostranstvoRegion', 'Godina', 'Mesec_proba', 'Grad', 'Nadkategorija']]
poslovi


# In[7]:


# Testiramo da li je ostalo oglasa koji nisu za inostranstvo
poslovi[(poslovi['ImaInostranstvoGrad'] != 'da') & (poslovi['ImaInostranstvoRegion'] != 'da')]


# In[8]:


# Provera postojanja NaN vrednosti 
poslovi.isna().sum()


# In[9]:


# Provera postojanja NULL vrednosti 
poslovi.info()


# In[10]:


# Da li nam je izbacio slučajeva gde su oba "inostranstvo" možemo da vidimo na kompaktniji način
poslovi[(poslovi['ImaInostranstvoGrad'] == 'da') & (poslovi['ImaInostranstvoRegion'] == 'da')].shape[0]


# In[11]:


# Izbacujemo broj konkurisanja i broj pregleda oglasa jednako 0
poslovi.drop(poslovi.loc[poslovi['BrojKonkurisanja']==0].index, inplace=True)
poslovi.drop(poslovi.loc[poslovi['BrojPregledaOglasa']==0].index, inplace=True)
poslovi


# In[12]:


# Izbacujemo specifične slučajeve kod načina konkurisanja (A i X)
poslovi = poslovi[(poslovi.NacinKonkurisanja != 'A') & (poslovi.NacinKonkurisanja != 'X')]
poslovi


# In[13]:


# Proveravamo da li je kod Načina konkurisanja ostalo nešto sa A i X
poslovi.NacinKonkurisanja.unique()


# In[14]:


# Bazični opis naših numeričkih varijabli koje ćemo gledati
poslovi.describe()


# ## *EDA*

# #### *EDA: Države*

# In[15]:


# Izbacivanje slučajeva gde je država 0 ili Rad na brodu
poslovi1 = poslovi[~poslovi['Drzava'].isin(["0","Rad na brodu"])]
poslovi1


# In[16]:


# Ovde gledam unique vrednosti kod Primarne kategorije
poslovi1.Drzava.unique()


# In[17]:


# Vizelizacija samo prvih 20
poslovi1['Drzava'].value_counts()[:20]


# In[18]:


# Vizelizacija samo prvih 10 (samo da vidimo šta se dešava, pre nego što nacrtamlo plotly grafik)
# poslovi1['Drzava'].value_counts().head(10).plot(kind='barh', figsize=(20,10))


# In[19]:


# Selektovanje zemalja koje mi trebaju za plotly (poseban df)
poslovi2 = poslovi1[poslovi1["Drzava"].isin(["Nemačka","Crna Gora", "Hrvatska", "Slovenija", "Rusija", "Mađarska", "Češka", "Bosna i Hercegovina", "Češka", "Ujedinjeni Arapski Emirati", "Slovačka"])]
poslovi2


# In[20]:


# Crtanje plotly grafika (koji su nam vizuelno lepši i idu na front page projekta)
fig = px.histogram(poslovi2, y="Drzava", nbins=20, orientation='h', color_discrete_sequence=['#518EBD'])
fig.update_layout(
    xaxis_title_text='Broj oglasa', # xaxis label
    yaxis_title_text='Država', # yaxis label
)
fig.show()


# #### *EDA: Godine*

# In[21]:


# Boxplot (da vidimo autlajere)
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.boxplot(x=poslovi['Godina'], color="darkblue")


# In[22]:


# Plotly
fig = px.histogram(poslovi, x="Godina", nbins=20, color_discrete_sequence=['#518EBD'])
fig.update_layout(
    xaxis_title_text='Godina', # xaxis label
    yaxis_title_text='Broj oglasa', # yaxis label
)
fig.show()


# #### *EDA: Broj konkurisanja*

# In[23]:


# Boxplot (da vidimo autlajere)
sns.boxplot(x=poslovi['BrojKonkurisanja'], color="darkblue")


# In[24]:


# Distribucija -- apsolutne vrednosti na y osi
sns.distplot(poslovi['BrojKonkurisanja'], color="darkblue", label="BrojKonkurisanja", kde=False)


# In[25]:


# Nećemo da izbacimo više od 5% podataka, pa da bismo bili sigurni gledamo percentile
for percentile in [0.95,0.96,0.97,0.98,0.99]:
    print(np.quantile(poslovi['BrojKonkurisanja'],percentile))


# **Note:** Uzeću granicu od 224 (97% percentil), dakle izbacujemo sve što je iznad toga (3% podataka).

# In[26]:


# Izbacujemo sve što izlazi iz tog opsega
poslovi = poslovi.drop(poslovi[poslovi.BrojKonkurisanja > 224].index)
poslovi


# In[27]:


# Distribucija nakon čišćenja (ovo je rađeno u Seabornu)
sns.distplot(poslovi['BrojKonkurisanja'], color="darkblue", label="BrojKonkurisanja", kde=False)


# #### *EDA: Broj konkurisanja (proba -- log transformacija)*

# In[28]:


# Log transformacija ove varijable, da vidimo da li je možda bolje da je tako uzimamo za modelovanje
poslovi['BrojKonkurisanjaLog'] = np.log(poslovi['BrojKonkurisanja'] + 1)
print(poslovi)
# Ovde je dodato + 1 da bismo izbegli -int kod vrednosti 0 (logaritam od 0 je problematika za sebe za regresiju)


# In[29]:


# Distribucija varijable koja je transformisana (log)
sns.distplot(poslovi['BrojKonkurisanjaLog'], color="darkblue", label="BrojKonkurisanjaLog", kde=False)


# #### *EDA: Broj pregleda oglasa*

# In[30]:


# Boxplot (da vidimo autlajere)
sns.boxplot(x=poslovi['BrojPregledaOglasa'], color="darkblue")


# In[31]:


# Određujemo limite za izbacivanje (uzeću najstrožiji kriterijum, zato što imamo grupisanje oko autlajera, a ne bih da poizbacujem toliko teksta )
for percentile in [0.95,0.96,0.97,0.98,0.99]:
    print(np.quantile(poslovi['BrojPregledaOglasa'],percentile))


# In[32]:


# # Izbacujemo sve što izlazi iz tog opsega
poslovi = poslovi.drop(poslovi[poslovi.BrojPregledaOglasa > 4962.82].index)
poslovi


# In[33]:


# Distribucija nakon čišćenja
sns.distplot(poslovi['BrojPregledaOglasa'], color="darkblue", label="BrojPregledaOglasa", kde=False)


# #### *EDA: Broj pregleda oglasa (proba -- log transformacija)*

# In[34]:


# Log transformacija ove varijable
poslovi['BrojPregledaOglasaLog'] = np.log(poslovi['BrojPregledaOglasa'] + 1)
print(poslovi)
# Ovde je dodato + 1 da bismo izbegli -int kod vrednosti 0 (logaritam od 0 je problematika za sebe za regresiju)


# In[35]:


# Distribucija varijable na novom setu podataka
sns.distplot(poslovi['BrojPregledaOglasaLog'], color="darkblue", label="BrojPregledaOglasaLog", kde=False)


# #### *EDA: Primarna kategorija (top 20)*

# In[36]:


# Ovde gledam unique vrednosti kod Primarne kategorije
poslovi.PrimarnaKategorija.unique()


# In[37]:


# Prebrajamo sve što imamo u Primarna kategorija
# Vizelizacija samo prvih 20
poslovi['PrimarnaKategorija'].value_counts()[:20]


# In[38]:


# Vizelizacija samo prvih 20
# poslovi['PrimarnaKategorija'].value_counts().head(20).plot(kind='barh', figsize=(20,10))


# In[39]:


# Kreiranje subseta sa top 20 kategorija
topi20 = poslovi['PrimarnaKategorija'].value_counts().head(20) 
topi20


# In[40]:


# Selektovanje primarnih kategorija u zaseban df za plotly
top20 = poslovi[poslovi["PrimarnaKategorija"].isin(["mašinstvo","građevina, geodezija", "IT", "elektrotehnika", "ugostiteljstvo", "priprema hrane", "zdravstvo", "transport", "trgovina, prodaja", "briga o lepoti"])]
# top20


# In[41]:


# Plotly
fig = px.histogram(top20, y="PrimarnaKategorija", nbins=20, orientation='h', color_discrete_sequence=['#518EBD'])
fig.update_layout(
    xaxis_title_text='Broj oglasa', # xaxis label
    yaxis_title_text='Primarna kategorija oglasa', # yaxis label
)
fig.show()


# #### *EDA: Standardizovana pozicija (top 20)*

# In[42]:


# Ovde gledam unique vrednosti kod Primarne kategorije
poslovi.StandardizovanaPozicija.unique()


# In[43]:


# Vizelizacija samo prvih 20
poslovi['StandardizovanaPozicija'].value_counts()[:20]


# In[44]:


# Vizelizacija samo prvih 20 (Seaborn)
# poslovi['StandardizovanaPozicija'].value_counts().head(20).plot(kind='barh', figsize=(20,10))


# In[45]:


# Selektovanje primarnih kategorija u zaseban df za plotly
top10srp = poslovi[poslovi["StandardizovanaPozicija"].isin(["vozač","kuvar", "građevinski inženjer", "konobar", "mašinski inženjer", "elektroinženjer", "električar", "lekar/doktor", "medicinska sestra/tehničar", "radnik u proizvodnji"])]
# top10srp


# In[46]:


# Plotly
fig = px.histogram(top10srp, y="StandardizovanaPozicija", nbins=20, orientation='h', color_discrete_sequence=['#518EBD'])
fig.update_layout(
    xaxis_title_text='Broj oglasa', # xaxis label
    yaxis_title_text='Standardizovana pozicija', # yaxis label
)
fig.show()


# #### *EDA: Natkategorija*

# In[47]:


# Ovde gledam unique vrednosti kod Primarne kategorije
poslovi.Nadkategorija.unique()


# In[48]:


# Vizelizacija samo prvih 20
poslovi['Nadkategorija'].value_counts()[:7]


# In[49]:


# Vizelizacija svih 7
poslovi['Nadkategorija'].value_counts().head(7).plot(kind='barh', figsize=(20,10))


# #### *EDA (ukrštanje): Broj konkurisanja po državama*

# In[50]:


# Sortiram da izvadim zemlje za koje imamo najveće vrednosti 
poslovi1.sort_values(by=['BrojKonkurisanja'], ascending=False)[:40]


# In[51]:


# Učitavanje posebnog fajla sa sumama po broju konkurisanja
suma = pd.read_csv('C:\\Users\\isidora.gataric\\Desktop\\UNDP\\2. Podaci\\2. Avgust_2020\\FINAL\\BrojKonk_PoDrzavama_Suma.csv', delimiter=';', encoding='utf-8')
suma


# In[52]:


# Pravimo df sa top 10 vrednosti
top10 = suma.nlargest(11, ['Broj konkurisanja (sum)']) 
top10


# In[53]:


# Crtanje grafika u plotly
fig = px.bar(top10, x="Broj konkurisanja (sum)", y="Drzava", orientation='h', color_discrete_sequence=['#518EBD'])
fig.update_layout(
    xaxis_title_text='Broj konkurisanja', # xaxis label
    yaxis_title_text='Država', # yaxis label
)
fig.show()


# #### *EDA (ukrštanje): Broj konkurisanja po natkategorijama*

# In[54]:


# Učitavanje posebnog fajla sa sumama broj konkurisanja po natkategorijama
natkat = pd.read_csv('C:\\Users\\isidora.gataric\\Desktop\\UNDP\\2. Podaci\\2. Avgust_2020\\FINAL\\brojkonk_po_natkat.csv', delimiter=';', encoding='utf-8')
# natkat


# In[55]:


# Crtanje grafika u plotly
fig = px.bar(natkat, x="BrojKonkurisanja", y="Natkategorija", orientation='h', color_discrete_sequence=['#518EBD'])
fig.update_layout(
    xaxis_title_text='Broj konkurisanja', # xaxis label
    yaxis_title_text='Kategorija oglasa', # yaxis label
)
fig.show()


# #### *EDA (ukrštanje): SRP-ovi po natkategorijama*

# In[56]:


# Pravimo subsets za svaku natkategoriju (7)
prva = poslovi.loc[poslovi['Nadkategorija'].isin(['Zdravstvo i farmacija'])]
druga = poslovi.loc[poslovi['Nadkategorija'].isin(['Turizam i ugostiteljstvo'])]
treca = poslovi.loc[poslovi['Nadkategorija'].isin(['Trgovina i usluge'])]
cetvrta = poslovi.loc[poslovi['Nadkategorija'].isin(['Tehničke nauke'])]
peta = poslovi.loc[poslovi['Nadkategorija'].isin(['Obrazovanje, umetnost i sport'])]
sesta = poslovi.loc[poslovi['Nadkategorija'].isin(['Informacione tehnologije'])]
sedma = poslovi.loc[poslovi['Nadkategorija'].isin(['Ekonomija'])]


# In[57]:


# Vizelizacija samo prvih 20 kod Zdravstva i farmacije
prva['StandardizovanaPozicija'].value_counts()[:20]


# In[58]:


# Vizelizacija samo prvih 20 kod Turizam i ugostiteljstvo
druga['StandardizovanaPozicija'].value_counts()[:20]


# In[59]:


# Vizelizacija samo prvih 20 kod Trgovina i usluge
treca['StandardizovanaPozicija'].value_counts()[:20]


# In[60]:


# Vizelizacija samo prvih 20 kod Tehničke nauke
cetvrta['StandardizovanaPozicija'].value_counts()[:20]


# In[61]:


# Vizelizacija samo prvih 20 kod Obrazovanje, umetnost, sport
peta['StandardizovanaPozicija'].value_counts()[:20]


# In[62]:


# Vizelizacija samo prvih 20 kod IT
sesta['StandardizovanaPozicija'].value_counts()[:20]


# In[63]:


# Vizelizacija samo prvih 20 kod Ekonomija
sedma['StandardizovanaPozicija'].value_counts()[:20]


# #### *EDA (ukrštanje): Primarna kategorija (top 3) po godinama*

# In[64]:


# Grupišemo Primarna kategorija po godinama, da vidimo šta se dešava
pd.get_dummies(poslovi, columns=['Godina']).groupby('PrimarnaKategorija').sum()


# In[65]:


# Selektujemo prva tri (najfrekventnije primarne kategorije)
prakse_prvi = poslovi.loc[poslovi['PrimarnaKategorija'].isin(['IT'])]
prakse_drugi = poslovi.loc[poslovi['PrimarnaKategorija'].isin(['mašinstvo'])]
prakse_treci = poslovi.loc[poslovi['PrimarnaKategorija'].isin(['građevina, geodezija'])]


# In[66]:


# Prikazujemo IT po godinama
sns.distplot(prakse_prvi['Godina'], color="blue", label="Godina", kde=False)


# In[67]:


# Prikazujemo mašinstvo po godinama
sns.distplot(prakse_drugi['Godina'], color="darkblue", label="Godina", kde=False)


# In[68]:


# Prikazujemo mašinstvo po godinama
sns.distplot(prakse_treci['Godina'], color="lightblue", label="Godina", kde=False)


# In[69]:


# Prikazujemo grafički preklopljeno sve tri kategorije
sns.distplot(prakse_prvi['Godina'], color="darkblue", label="Godina", kde=False)
sns.distplot(prakse_drugi['Godina'], color="blue", label="Godina", kde=False)
sns.distplot(prakse_treci['Godina'], color="lightblue", label="Godina", kde=False)


# #### *EDA (ukrštanje): Država (neke za koje mislimo da su top 3) po godinama*

# In[70]:


# Grupišemo Primarna kategorija po godinama, da vidimo šta se dešava
# Koristimo poslovi1 zato što tu imamo očišćenu varijablu država (izbačeno 0 i Rad na brodu)
pd.get_dummies(poslovi1, columns=['Godina']).groupby('Drzava').sum()


# In[71]:


# Selektujemo tri države za koje mislimo da su jako popularne
kec = poslovi1.loc[poslovi1['Drzava'].isin(['Nemačka'])]
dvojka = poslovi1.loc[poslovi1['Drzava'].isin(['Austrija'])]
trojka = poslovi1.loc[poslovi1['Drzava'].isin(['Švajcarska'])]


# In[72]:


# Prikazujemo Nemačka po godinama
sns.distplot(kec['Godina'], color="darkblue", label="Godina", kde=False)


# In[73]:


# Prikazujemo Austrija po godinama
sns.distplot(dvojka['Godina'], color="blue", label="Godina", kde=False)


# In[74]:


# Prikazujemo Švajcarska po godinama
sns.distplot(trojka['Godina'], color="lightblue", label="Godina", kde=False)


# In[75]:


# Prikazujemo grafički preklopljeno sve tri kategorije
sns.distplot(kec['Godina'], color="darkblue", label="Godina", kde=False)
sns.distplot(dvojka['Godina'], color="blue", label="Godina", kde=False)
sns.distplot(trojka['Godina'], color="lightblue", label="Godina", kde=False)


# #### *EDA (ukrštanje): Natkategorije po godinama*

# In[76]:


# Grupišemo po godinama
pd.get_dummies(poslovi, columns=['Godina']).groupby('Nadkategorija').sum()


# In[77]:


# Pravimo subsets da bismo mogli da vizuelno prikažemo to sve lepo
prva = poslovi.loc[poslovi['Nadkategorija'].isin(['Zdravstvo i farmacija'])]
druga = poslovi.loc[poslovi['Nadkategorija'].isin(['Turizam i ugostiteljstvo'])]
treca = poslovi.loc[poslovi['Nadkategorija'].isin(['Trgovina i usluge'])]
cetvrta = poslovi.loc[poslovi['Nadkategorija'].isin(['Tehničke nauke'])]
peta = poslovi.loc[poslovi['Nadkategorija'].isin(['Obrazovanje, umetnost i sport'])]
sesta = poslovi.loc[poslovi['Nadkategorija'].isin(['Informacione tehnologije'])]
sedma = poslovi.loc[poslovi['Nadkategorija'].isin(['Ekonomija'])]


# In[78]:


# Zdravstvo i farmacija po godinama
sns.distplot(prva['Godina'], color="darkblue", label="Godina", kde=False)


# In[79]:


# Turizam i ugostiteljstvo po godinama
sns.distplot(druga['Godina'], color="darkblue", label="Godina", kde=False)


# In[80]:


# Trgovina i usluge po godinama
sns.distplot(treca['Godina'], color="darkblue", label="Godina", kde=False)


# In[81]:


# Tehničke nauke po godinama
sns.distplot(cetvrta['Godina'], color="darkblue", label="Godina", kde=False)


# In[82]:


# Obrazovanje, umetnost i sport po godinama
sns.distplot(peta['Godina'], color="darkblue", label="Godina", kde=False)


# In[83]:


# IT po godinama
sns.distplot(sesta['Godina'], color="darkblue", label="Godina", kde=False)


# In[84]:


# Ekonomija po godinama
sns.distplot(sedma['Godina'], color="darkblue", label="Godina", kde=False)


# In[85]:


# Mariji za front page
# sns.set(rc={'figure.figsize':(16,10)})
# sns.barplot(x="Nadkategorija", y="BrojKonkurisanja", data=poslovi)


# #### *EDA (ukrštanje): Broj konkurisanja po godinama*

# In[86]:


# Sumiranje broja konkurisanja po godinama
#probica = poslovi.groupby(['Godina']).sum()
#probica


# In[87]:


# Crtanje grafika za Plotly
plo = pd.read_csv('C:\\Users\\isidora.gataric\\Desktop\\UNDP\\2. Podaci\\2. Avgust_2020\\FINAL\\plotly_broj_konk_god.csv', delimiter=';', encoding='utf-8')
# plo


# In[88]:


# Plotly
fig = px.line(plo, x="Godina", y="BrojKonkurisanja", color_discrete_sequence=['#518EBD'])
fig.update_layout(
    xaxis_title_text='Godina', # xaxis label
    yaxis_title_text='Broj konkurisanja', # yaxis label
)
fig.show()


# #### *EDA (ukrštanje): Broj konkurisanja po mesecima*

# In[89]:


# Kako se po mesecima kroz godine ponašanju obe varijable
tri = poslovi.loc[poslovi['Godina'].isin(['2013'])]
cetiri = poslovi.loc[poslovi['Godina'].isin(['2014'])]
pet = poslovi.loc[poslovi['Godina'].isin(['2015'])]
sest = poslovi.loc[poslovi['Godina'].isin(['2016'])]
sedam = poslovi.loc[poslovi['Godina'].isin(['2017'])]
osam = poslovi.loc[poslovi['Godina'].isin(['2018'])]
devet = poslovi.loc[poslovi['Godina'].isin(['2019'])]
deset = poslovi.loc[poslovi['Godina'].isin(['2020'])]


# In[90]:


# Godina 2013 po mesecima Broj Konkurisanja
ax3 = sns.lineplot(x="Mesec_proba", y="BrojKonkurisanja", ci=None, data=tri)


# In[91]:


# Godina 2014 po mesecima Broj Konkurisanja
ax4 = sns.lineplot(x="Mesec_proba", y="BrojKonkurisanja", ci=None, data=cetiri)


# In[92]:


# Godina 2015 po mesecima Broj Konkurisanja
ax5 = sns.lineplot(x="Mesec_proba", y="BrojKonkurisanja", ci=None, data=pet)


# In[93]:


# Godina 2016 po mesecima Broj Konkurisanja
ax6 = sns.lineplot(x="Mesec_proba", y="BrojKonkurisanja", ci=None, data=sest)


# In[94]:


# Godina 2017 po mesecima Broj Konkurisanja
ax7 = sns.lineplot(x="Mesec_proba", y="BrojKonkurisanja", ci=None, data=sedam)


# In[95]:


# Godina 2018 po mesecima Broj Konkurisanja
ax8 = sns.lineplot(x="Mesec_proba", y="BrojKonkurisanja", ci=None, data=osam)


# In[96]:


# Godina 2019 po mesecima Broj Konkurisanja
ax9 = sns.lineplot(x="Mesec_proba", y="BrojKonkurisanja", ci=None, data=devet)


# In[97]:


# Godina 2020 po mesecima Broj Konkurisanja
ax10 = sns.lineplot(x="Mesec_proba", y="BrojKonkurisanja", ci=None, data=deset)


# #### *EDA (ukrštanje): Broj konkurisanja po natkategorijama*

# In[98]:


# Pravimo subsets da bismo mogli da vizuelno prikažemo to sve lepo
prva = poslovi.loc[poslovi['Nadkategorija'].isin(['Zdravstvo i farmacija'])]
druga = poslovi.loc[poslovi['Nadkategorija'].isin(['Turizam i ugostiteljstvo'])]
treca = poslovi.loc[poslovi['Nadkategorija'].isin(['Trgovina i usluge'])]
cetvrta = poslovi.loc[poslovi['Nadkategorija'].isin(['Tehničke nauke'])]
peta = poslovi.loc[poslovi['Nadkategorija'].isin(['Obrazovanje, umetnost i sport'])]
sesta = poslovi.loc[poslovi['Nadkategorija'].isin(['Informacione tehnologije'])]
sedma = poslovi.loc[poslovi['Nadkategorija'].isin(['Ekonomija'])]


# In[99]:


# Broj konkurisanja kroz godine u Zdravtsvo i farmacija
ax8 = sns.lineplot(x="Godina", y="BrojKonkurisanja", ci=None, data=prva)


# In[100]:


# Broj konkurisanja kroz godine u Turizam i ugostiteljstvo
ax8 = sns.lineplot(x="Godina", y="BrojKonkurisanja", ci=None, data=druga)


# In[101]:


# Broj konkurisanja kroz godine u Trgovina i usluge
ax8 = sns.lineplot(x="Godina", y="BrojKonkurisanja", ci=None, data=treca)


# In[102]:


# Broj konkurisanja kroz godine u Tehničke nauke
ax8 = sns.lineplot(x="Godina", y="BrojKonkurisanja", ci=None, data=cetvrta)


# In[103]:


# Broj konkurisanja kroz godine u Obrazovanje, umetnost, sport
ax8 = sns.lineplot(x="Godina", y="BrojKonkurisanja", ci=None, data=peta)


# In[104]:


# Broj konkurisanja kroz godine u IT
ax8 = sns.lineplot(x="Godina", y="BrojKonkurisanja", ci=None, data=sesta)


# In[105]:


# Broj konkurisanja kroz godine u Ekonomija
ax8 = sns.lineplot(x="Godina", y="BrojKonkurisanja", ci=None, data=sedma)


# #### *EDA (ukrštanje): Broj pregleda oglasa po godinama*

# In[106]:


# Crtanje grafika za Plotly
pog = pd.read_csv('C:\\Users\\isidora.gataric\\Desktop\\UNDP\\2. Podaci\\2. Avgust_2020\\FINAL\\brojpreg_po_god.csv', delimiter=';', encoding='utf-8')
# pog


# In[107]:


# Plotly
fig = px.line(pog, x="Natkategorija", y="BrojPregledaOglasa", color_discrete_sequence=['#518EBD'])
fig.update_layout(
    xaxis_title_text='Godina', # xaxis label
    yaxis_title_text='Broj pregleda oglasa', # yaxis label
)
fig.show()


# #### *EDA (ukrštanje): Broj pregleda oglasa po mesecima*

# In[108]:


# Kako se po mesecima kroz godine ponašanju obe varijable
tri = poslovi.loc[poslovi['Godina'].isin(['2013'])]
cetiri = poslovi.loc[poslovi['Godina'].isin(['2014'])]
pet = poslovi.loc[poslovi['Godina'].isin(['2015'])]
sest = poslovi.loc[poslovi['Godina'].isin(['2016'])]
sedam = poslovi.loc[poslovi['Godina'].isin(['2017'])]
osam = poslovi.loc[poslovi['Godina'].isin(['2018'])]
devet = poslovi.loc[poslovi['Godina'].isin(['2019'])]
deset = poslovi.loc[poslovi['Godina'].isin(['2020'])]


# In[109]:


# Godina 2013 po mesecima Broj pregleda oglasa
ax3 = sns.lineplot(x="Mesec_proba", y="BrojPregledaOglasa", ci=None, data=tri)


# In[110]:


# Godina 2014 po mesecima Broj pregleda oglasa
ax4 = sns.lineplot(x="Mesec_proba", y="BrojPregledaOglasa", ci=None, data=cetiri)


# In[111]:


# Godina 2015 po mesecima Broj pregleda oglasa
ax5 = sns.lineplot(x="Mesec_proba", y="BrojPregledaOglasa", ci=None, data=pet)


# In[112]:


# Godina 2016 po mesecima Broj pregleda oglasa
ax6 = sns.lineplot(x="Mesec_proba", y="BrojPregledaOglasa", ci=None, data=sest)


# In[113]:


# Godina 2017 po mesecima Broj pregleda oglasa
ax7 = sns.lineplot(x="Mesec_proba", y="BrojPregledaOglasa", ci=None, data=sedam)


# In[114]:


# Godina 2018 po mesecima Broj pregleda oglasa
ax8 = sns.lineplot(x="Mesec_proba", y="BrojPregledaOglasa", ci=None, data=osam)


# In[115]:


# Godina 2019 po mesecima Broj pregleda oglasa
ax9 = sns.lineplot(x="Mesec_proba", y="BrojPregledaOglasa", ci=None, data=devet)


# In[116]:


# Godina 2020 po mesecima Broj pregleda oglasa
ax10 = sns.lineplot(x="Mesec_proba", y="BrojPregledaOglasa", ci=None, data=deset)


# #### *EDA (ukrštanje): Broj pregleda oglasa po natkategorijama*

# In[117]:


# Pravimo subsets da bismo mogli da vizuelno prikažemo to sve lepo
prva = poslovi.loc[poslovi['Nadkategorija'].isin(['Zdravstvo i farmacija'])]
druga = poslovi.loc[poslovi['Nadkategorija'].isin(['Turizam i ugostiteljstvo'])]
treca = poslovi.loc[poslovi['Nadkategorija'].isin(['Trgovina i usluge'])]
cetvrta = poslovi.loc[poslovi['Nadkategorija'].isin(['Tehničke nauke'])]
peta = poslovi.loc[poslovi['Nadkategorija'].isin(['Obrazovanje, umetnost i sport'])]
sesta = poslovi.loc[poslovi['Nadkategorija'].isin(['Informacione tehnologije'])]
sedma = poslovi.loc[poslovi['Nadkategorija'].isin(['Ekonomija'])]


# In[118]:


# Broj pregleda oglasa kroz godine u Zdravtsvo i farmacija
ax10 = sns.lineplot(x="Godina", y="BrojPregledaOglasa", ci=None, data=prva)


# In[119]:


# Broj pregleda oglasa kroz godine u Turizam i ugostiteljstvo
ax10 = sns.lineplot(x="Godina", y="BrojPregledaOglasa", ci=None, data=druga)


# In[120]:


# Broj pregleda oglasa kroz godine u Trgovina i usluge
ax10 = sns.lineplot(x="Godina", y="BrojPregledaOglasa", ci=None, data=treca)


# In[121]:


# Broj pregleda oglasa kroz godine u Tehničke nauke
ax10 = sns.lineplot(x="Godina", y="BrojPregledaOglasa", ci=None, data=cetvrta)


# In[122]:


# Broj pregleda oglasa kroz godine u Obrazovanje, umetnost, sport
ax10 = sns.lineplot(x="Godina", y="BrojPregledaOglasa", ci=None, data=peta)


# In[123]:


# Broj pregleda oglasa kroz godine u IT
ax10 = sns.lineplot(x="Godina", y="BrojPregledaOglasa", ci=None, data=sesta)


# In[124]:


# Broj pregleda oglasa kroz godine u Ekonomija
ax10 = sns.lineplot(x="Godina", y="BrojPregledaOglasa", ci=None, data=sedma)


# #### *EDA: Korelacija (pre Regresione analize - ML)*

# In[125]:


# Pirsonova korelacija (mada ovo nije najbolji izbor, probaću i sa Spirman i Kandal-tau)
korelacija = poslovi[['BrojKonkurisanja','BrojPregledaOglasa']]
korelacija.corr(method='pearson', min_periods=1)


# In[126]:


# Kendal-tau (varijable nemaju Gausovu raspodelu)
korelacija.corr(method='kendall', min_periods=1)


# In[127]:


# Spirman (varijable nemaju Gausovu raspodelu)
korelacija.corr(method='spearman', min_periods=1)


# ## *ML -- Regresiona analiza*

# ## *Linearna regresija po natkategorijama/godinama*

# #### *IQR po natkategorijama*

# In[128]:


# Definisanje fukncije za IQR (treba nam strožiji način prečišćavanja podataka, pa smo zato i ovo odradili)
def ciscenje (datacolumn):
    sorted(datacolumn)
    Q1,Q3 = np.percentile(datacolumn, [25,75])
    IQR = Q3 - Q1
    locmin = Q1 - (1.5 * IQR)
    locmax = Q3 + (1.5 * IQR)
    return locmin, locmax


# In[129]:


# Čišćenje lokalnih min/max vrednosti
kategorije=poslovi.copy()
for y in list (dict.fromkeys(kategorije['Nadkategorija'].values)):
    for m in ['BrojKonkurisanja', 'BrojPregledaOglasa']:
        d=kategorije[(kategorije['Nadkategorija']==y)][m]
        if d.size>0:
            locmin, locmax=ciscenje(d)
            kategorije.drop(index=kategorije.index[(kategorije['Nadkategorija']==y)&(kategorije[m]<locmin)], inplace=True)
            kategorije.drop(index=kategorije.index[(kategorije['Nadkategorija']==y)&(kategorije[m]>locmax)], inplace=True)              


# In[130]:


# Provera kako to izgleda nakon čišćenja
kategorije.shape


# ### *Linearna regresija (standardna) po natkategorijama*

# #### *LR (standardna) -- Zdravstvo i farmacija*

# In[131]:


# Regresija (skater)
df1= kategorije.loc[kategorije['Nadkategorija'].isin(['Zdravstvo i farmacija'])]
X = df1.iloc[:, 2].values.reshape(-1, 1)
Y = df1.iloc[:, 1].values.reshape(-1, 1)
linear_regressor = LinearRegression()
r = linear_regressor.fit(X, Y)
Y_pred = linear_regressor.predict(X)

plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()


# In[132]:


# Regresioni skor
r.score(X,Y)


# In[133]:


# Regresioni koeficijent
r.coef_


# #### *LR (standardna) -- Turizam i ugostiteljstvo*

# In[134]:


# Regresija (skater)
df2= kategorije.loc[kategorije['Nadkategorija'].isin(['Turizam i ugostiteljstvo'])]
X = df2.iloc[:, 2].values.reshape(-1, 1)
Y = df2.iloc[:, 1].values.reshape(-1, 1)
linear_regressor = LinearRegression()
r = linear_regressor.fit(X, Y)
Y_pred = linear_regressor.predict(X)

plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()


# In[135]:


# Regresioni skor
r.score(X,Y)


# In[136]:


# Regresioni koeficijent
r.coef_


# #### *LR (standardna) -- Trgovina u usluge*

# In[137]:


# Regresija (skater)
df3= kategorije.loc[kategorije['Nadkategorija'].isin(['Trgovina i usluge'])]
X = df3.iloc[:, 2].values.reshape(-1, 1)
Y = df3.iloc[:, 1].values.reshape(-1, 1)
linear_regressor = LinearRegression()
r = linear_regressor.fit(X, Y)
Y_pred = linear_regressor.predict(X)

plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()


# In[138]:


# Regresioni skor
r.score(X,Y)


# In[139]:


# Regresioni koeficijent
r.coef_


# #### *LR (standardna) -- Tehničke nauke*

# In[140]:


# Regresija (skater)
df4= kategorije.loc[kategorije['Nadkategorija'].isin(['Tehničke nauke'])]
X = df4.iloc[:, 2].values.reshape(-1, 1)
Y = df4.iloc[:, 1].values.reshape(-1, 1)
linear_regressor = LinearRegression()
r = linear_regressor.fit(X, Y)
Y_pred = linear_regressor.predict(X)

plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()


# In[141]:


# Regresioni skor
r.score(X,Y)


# In[142]:


# Regresioni koeficijent
r.coef_


# #### *LR (standardna) -- Obrazovanje, umetnost, nauka*

# In[143]:


# Regresija (skater)
df5= kategorije.loc[kategorije['Nadkategorija'].isin(['Obrazovanje, umetnost i sport'])]
X = df5.iloc[:, 2].values.reshape(-1, 1)
Y = df5.iloc[:, 1].values.reshape(-1, 1)
linear_regressor = LinearRegression()
r = linear_regressor.fit(X, Y)
Y_pred = linear_regressor.predict(X)

plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()


# In[144]:


# Regresioni skor
r.score(X,Y)


# In[145]:


# Regresioni koeficijent
r.coef_


# #### *LR (standardna) -- IT*

# In[146]:


# Regresija (skater)
df6= kategorije.loc[kategorije['Nadkategorija'].isin(['Informacione tehnologije'])]
X = df6.iloc[:, 2].values.reshape(-1, 1)
Y = df6.iloc[:, 1].values.reshape(-1, 1)
linear_regressor = LinearRegression()
r = linear_regressor.fit(X, Y)
Y_pred = linear_regressor.predict(X)

plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()


# In[147]:


# Regresioni skor
r.score(X,Y)


# In[148]:


# Regresioni koeficijent
r.coef_


# #### *LR (standardna) -- Ekonomija*

# In[149]:


# Regresija (skater)
df7= kategorije.loc[kategorije['Nadkategorija'].isin(['Ekonomija'])]
X = df7.iloc[:, 2].values.reshape(-1, 1)
Y = df7.iloc[:, 1].values.reshape(-1, 1)
linear_regressor = LinearRegression()
r = linear_regressor.fit(X, Y)
Y_pred = linear_regressor.predict(X)

plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()


# In[150]:


# Regresioni skor
r.score(X,Y)


# In[151]:


# Regresioni koeficijent
r.coef_


# #### *IQR po godinama*

# In[152]:


# Definisanje fukncije za IQR
def ciscenje (datacolumn):
    sorted(datacolumn)
    Q1,Q3 = np.percentile(datacolumn, [25,75])
    IQR = Q3 - Q1
    locmin = Q1 - (1.5 * IQR)
    locmax = Q3 + (1.5 * IQR)
    return locmin, locmax


# In[153]:


# Čišćenje lokalnih min/max vrednosti
kategorije1=poslovi.copy()
for y in list (dict.fromkeys(kategorije1['Godina'].values)):
    for m in ['BrojKonkurisanja', 'BrojPregledaOglasa']:
        d=kategorije1[(kategorije1['Godina']==y)][m]
        if d.size>0:
            locmin, locmax=ciscenje(d)
            kategorije1.drop(index=kategorije1.index[(kategorije1['Godina']==y)&(kategorije1[m]<locmin)], inplace=True)
            kategorije1.drop(index=kategorije1.index[(kategorije1['Godina']==y)&(kategorije1[m]>locmax)], inplace=True)                


# In[154]:


# Provera kako to izgleda nakon čišćenja
kategorije1.shape


# ### *Linearna regresija (standardna) po godinama*

# #### *LR (standardna) -- 2013*

# In[155]:


# Regresija (skater)
df8= kategorije1.loc[kategorije1['Godina'].isin(['2013'])]
X = df8.iloc[:, 2].values.reshape(-1, 1)
Y = df8.iloc[:, 1].values.reshape(-1, 1)
linear_regressor = LinearRegression()
r = linear_regressor.fit(X, Y)
Y_pred = linear_regressor.predict(X)

plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()


# In[156]:


# Regresioni skor
r.score(X,Y)


# In[157]:


# Regresioni koeficijent
r.coef_


# #### *LR (standardna) -- 2014*

# In[158]:


# Regresija (skater)
df9= kategorije1.loc[kategorije1['Godina'].isin(['2014'])]
X = df9.iloc[:, 2].values.reshape(-1, 1)
Y = df9.iloc[:, 1].values.reshape(-1, 1)
linear_regressor = LinearRegression()
r = linear_regressor.fit(X, Y)
Y_pred = linear_regressor.predict(X)

plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()


# In[159]:


# Regresioni skor
r.score(X,Y)


# In[160]:


# Regresioni koeficijent
r.coef_


# #### *LR (standardna) -- 2015*

# In[161]:


# Regresija (skater)
df10= kategorije1.loc[kategorije1['Godina'].isin(['2015'])]
X = df10.iloc[:, 2].values.reshape(-1, 1)
Y = df10.iloc[:, 1].values.reshape(-1, 1)
linear_regressor = LinearRegression()
r = linear_regressor.fit(X, Y)
Y_pred = linear_regressor.predict(X)

plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()


# In[162]:


# Regresioni skor
r.score(X,Y)


# In[163]:


# Regresioni koeficijent
r.coef_


# #### *LR (standardna) -- 2016*

# In[164]:


# Regresija (skater)
df11= kategorije1.loc[kategorije1['Godina'].isin(['2016'])]
X = df11.iloc[:, 2].values.reshape(-1, 1)
Y = df11.iloc[:, 1].values.reshape(-1, 1)
linear_regressor = LinearRegression()
r = linear_regressor.fit(X, Y)
Y_pred = linear_regressor.predict(X)

plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()


# In[165]:


# Regresioni skor
r.score(X,Y)


# In[166]:


# Regresioni koeficijent
r.coef_


# #### *LR (standardna) -- 2017*

# In[167]:


# Regresija (skater)
df12= kategorije1.loc[kategorije1['Godina'].isin(['2017'])]
X = df12.iloc[:, 2].values.reshape(-1, 1)
Y = df12.iloc[:, 1].values.reshape(-1, 1)
linear_regressor = LinearRegression()
r = linear_regressor.fit(X, Y)
Y_pred = linear_regressor.predict(X)

plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()


# In[168]:


# Regresioni skor
r.score(X,Y)


# In[169]:


# Regresioni koeficijent
r.coef_


# #### *LR (standardna) -- 2018*

# In[170]:


# Regresija (skater)
df13= kategorije1.loc[kategorije1['Godina'].isin(['2018'])]
X = df13.iloc[:, 2].values.reshape(-1, 1)
Y = df13.iloc[:, 1].values.reshape(-1, 1)
linear_regressor = LinearRegression()
r = linear_regressor.fit(X, Y)
Y_pred = linear_regressor.predict(X)

plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()


# In[171]:


# Regresioni skor
r.score(X,Y)


# In[172]:


# Regresioni koeficijent
r.coef_


# #### *LR (standardna) -- 2019*

# In[173]:


# Regresija (skater)
df14= kategorije1.loc[kategorije1['Godina'].isin(['2019'])]
X = df14.iloc[:, 2].values.reshape(-1, 1)
Y = df14.iloc[:, 1].values.reshape(-1, 1)
linear_regressor = LinearRegression()
r = linear_regressor.fit(X, Y)
Y_pred = linear_regressor.predict(X)

plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()


# In[174]:


# Regresioni skor
r.score(X,Y)


# In[175]:


# Regresioni koeficijent
r.coef_


# #### *LR (standardna) -- 2020*

# In[176]:


# Regresija (skater)
df15= kategorije1.loc[kategorije1['Godina'].isin(['2020'])]
X = df15.iloc[:, 2].values.reshape(-1, 1)
Y = df15.iloc[:, 1].values.reshape(-1, 1)
linear_regressor = LinearRegression()
r = linear_regressor.fit(X, Y)
Y_pred = linear_regressor.predict(X)

plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()


# In[177]:


# Regresioni skor
r.score(X,Y)


# In[178]:


# Regresioni koeficijent
r.coef_


# ## *Linearna regresija (modifikovana -- median/round) po natkategorijama/godinama*

# #### *Transformacija postojećih varijable u modifikovane (median/round)*

# In[179]:


# Prvo treba napraviti te varijable u setu koji vec imamo, krecemo od seta kategorije (round)
kategorije['BrojPregledaRounded']=kategorije['BrojPregledaOglasa'].round(-2)
kategorije


# In[180]:


# Median za BrojKonkurisanja
df = pd.DataFrame(columns=['Nadkategorija','BrojPregledaRounded','size','BrojKonkurisanjaMedian'])


# In[181]:


# Upisujemo vrednosti u dve nove varijable Round/Median
for n in list(dict.fromkeys(kategorije['Nadkategorija'].values)):
    for r in list(dict.fromkeys(kategorije['BrojPregledaRounded'].values)):
        extr_app=[]
        extr_app.append(n)
        extr_app.append(r)
        d=kategorije[(kategorije['Nadkategorija']==n)& (kategorije['BrojPregledaRounded']==r)]['BrojKonkurisanja']
#       print(d.size)
        if d.size>8:
            extr_app.append(d.size)
            extr_app.append(statistics.median(d.tolist()))
#            print(extr_app)
            df=df.append(pd.Series(extr_app,index=df.columns), ignore_index=True)
df


# #### *LR (modifikovana) -- Zdravstvo i farmacija*

# In[182]:


# Regresija (skater)
df1= df.loc[df['Nadkategorija'].isin(['Zdravstvo i farmacija'])]
X = df1.iloc[:, 1].values.reshape(-1, 1)
Y = df1.iloc[:, 3].values.reshape(-1, 1)
linear_regressor = LinearRegression()
r = linear_regressor.fit(X, Y)
Y_pred = linear_regressor.predict(X)

plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()


# In[183]:


# Regresioni skor
r.score(X,Y)


# In[184]:


# Regresioni koeficijent
r.coef_


# #### *LR (modifikovana) -- Turizam i ugostiteljstvo*

# In[185]:


# Regresija (skater)
df2= df.loc[df['Nadkategorija'].isin(['Turizam i ugostiteljstvo'])]
X = df2.iloc[:, 1].values.reshape(-1, 1)
Y = df2.iloc[:, 3].values.reshape(-1, 1)
linear_regressor = LinearRegression()
r = linear_regressor.fit(X, Y)
Y_pred = linear_regressor.predict(X)

plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()


# In[186]:


# Regresioni skor
r.score(X,Y)


# In[187]:


# Regresioni koeficijent
r.coef_


# #### *LR (modifikovana) -- Trgovina i usluge*

# In[188]:


# Regresija (skater)
df3= df.loc[df['Nadkategorija'].isin(['Trgovina i usluge'])]
X = df3.iloc[:, 1].values.reshape(-1, 1)
Y = df3.iloc[:, 3].values.reshape(-1, 1)
linear_regressor = LinearRegression()
r = linear_regressor.fit(X, Y)
Y_pred = linear_regressor.predict(X)

plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()


# In[189]:


# Regresioni skor
r.score(X,Y)


# In[190]:


# Regresioni koeficijent
r.coef_


# #### *LR (modifikovana) -- Tehničke nauke*

# In[191]:


# Regresija (skater)
df4= df.loc[df['Nadkategorija'].isin(['Tehničke nauke'])]
X = df4.iloc[:, 1].values.reshape(-1, 1)
Y = df4.iloc[:, 3].values.reshape(-1, 1)
linear_regressor = LinearRegression()
r = linear_regressor.fit(X, Y)
Y_pred = linear_regressor.predict(X)

plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()


# In[192]:


# Regresioni skor
r.score(X,Y)


# In[193]:


# Regresioni koeficijent
r.coef_


# #### *LR (modifikovana) -- Obrazovanje, umetnost, nauka*

# In[194]:


# Regresija (skater)
df5= df.loc[df['Nadkategorija'].isin(['Obrazovanje, umetnost i sport'])]
X = df5.iloc[:, 1].values.reshape(-1, 1)
Y = df5.iloc[:, 3].values.reshape(-1, 1)
linear_regressor = LinearRegression()
r = linear_regressor.fit(X, Y)
Y_pred = linear_regressor.predict(X)

plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()


# In[195]:


# Regresioni skor
r.score(X,Y)


# In[196]:


# Regresioni koeficijent
r.coef_


# #### *LR (modifikovana) -- IT*

# In[197]:


# Regresija (skater)
df6= df.loc[df['Nadkategorija'].isin(['Informacione tehnologije'])]
X = df6.iloc[:, 1].values.reshape(-1, 1)
Y = df6.iloc[:, 3].values.reshape(-1, 1)
linear_regressor = LinearRegression()
r = linear_regressor.fit(X, Y)
Y_pred = linear_regressor.predict(X)

plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()


# In[198]:


# Regresioni skor
r.score(X,Y)


# In[199]:


# Regresioni koeficijent
r.coef_


# #### *LR (modifikovana) -- Ekonomija*

# In[200]:


# Regresija (skater)
df7= df.loc[df['Nadkategorija'].isin(['Ekonomija'])]
X = df7.iloc[:, 1].values.reshape(-1, 1)
Y = df7.iloc[:, 3].values.reshape(-1, 1)
linear_regressor = LinearRegression()
r = linear_regressor.fit(X, Y)
Y_pred = linear_regressor.predict(X)

plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()


# In[201]:


# Regresioni skor
r.score(X,Y)


# In[202]:


# Regresioni koeficijent
r.coef_


# #### *Transformacija postojećih varijable u modifikovane (median/round) -- Godine*

# In[203]:


# Median za BrojKonkurisanja
df1 = pd.DataFrame(columns=['Godina','BrojPregledaRounded','size','BrojKonkurisanjaMedian'])


# In[204]:


# Upisujemo vrednosti u dve nove varijable Round/Median
for n in list(dict.fromkeys(kategorije['Godina'].values)):
    for r in list(dict.fromkeys(kategorije['BrojPregledaRounded'].values)):
        extr_app=[]
        extr_app.append(n)
        extr_app.append(r)
        d=kategorije[(kategorije['Godina']==n)& (kategorije['BrojPregledaRounded']==r)]['BrojKonkurisanja']
#       print(d.size)
        if d.size>8:
            extr_app.append(d.size)
            extr_app.append(statistics.median(d.tolist()))
#            print(extr_app)
            df1=df1.append(pd.Series(extr_app,index=df1.columns), ignore_index=True)
df1


# #### *LR (modifikovana) -- 2013*

# In[205]:


# Regresija (skater)
df8= df1.loc[df1['Godina'].isin([2013])]
#df8 = df1[df1['Godina']=='2016']]
X = df8.iloc[:, 1].values.reshape(-1, 1)
Y = df8.iloc[:, 3].values.reshape(-1, 1)
linear_regressor = LinearRegression()
r = linear_regressor.fit(X, Y)
Y_pred = linear_regressor.predict(X)

plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()


# In[206]:


# Regresioni skor
r.score(X,Y)


# In[207]:


# Regresioni koeficijent
r.coef_


# #### *LR (modifikovana) -- 2014*

# In[208]:


# Regresija (skater)
df9= df1.loc[df1['Godina'].isin([2014])]
X = df9.iloc[:, 1].values.reshape(-1, 1)
Y = df9.iloc[:, 3].values.reshape(-1, 1)
linear_regressor = LinearRegression()
r = linear_regressor.fit(X, Y)
Y_pred = linear_regressor.predict(X)

plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()


# In[209]:


# Regresioni skor
r.score(X,Y)


# In[210]:


# Regresioni koeficijent
r.coef_


# #### *LR (modifikovana) -- 2015*

# In[211]:


# Regresija (skater)
df10= df1.loc[df1['Godina'].isin([2015])]
X = df10.iloc[:, 1].values.reshape(-1, 1)
Y = df10.iloc[:, 3].values.reshape(-1, 1)
linear_regressor = LinearRegression()
r = linear_regressor.fit(X, Y)
Y_pred = linear_regressor.predict(X)

plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()


# In[212]:


# Regresioni skor
r.score(X,Y)


# In[213]:


# Regresioni koeficijent
r.coef_


# #### *LR (modifikovana) -- 2016*

# In[214]:


# Regresija (skater)
df11= df1.loc[df1['Godina'].isin([2016])]
X = df11.iloc[:, 1].values.reshape(-1, 1)
Y = df11.iloc[:, 3].values.reshape(-1, 1)
linear_regressor = LinearRegression()
r = linear_regressor.fit(X, Y)
Y_pred = linear_regressor.predict(X)

plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()


# In[215]:


# Regresioni skor
r.score(X,Y)


# In[216]:


# Regresioni koeficijent
r.coef_


# #### *LR (modifikovana) -- 2017*

# In[217]:


# Regresija (skater)
df12= df1.loc[df1['Godina'].isin([2017])]
X = df12.iloc[:, 1].values.reshape(-1, 1)
Y = df12.iloc[:, 3].values.reshape(-1, 1)
linear_regressor = LinearRegression()
r = linear_regressor.fit(X, Y)
Y_pred = linear_regressor.predict(X)

plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()


# In[218]:


# Regresioni skor
r.score(X,Y)


# In[219]:


# Regresioni koeficijent
r.coef_


# #### *LR (modifikovana) -- 2018*

# In[220]:


# Regresija (skater)
df13= df1.loc[df1['Godina'].isin([2018])]
X = df13.iloc[:, 1].values.reshape(-1, 1)
Y = df13.iloc[:, 3].values.reshape(-1, 1)
linear_regressor = LinearRegression()
r = linear_regressor.fit(X, Y)
Y_pred = linear_regressor.predict(X)

plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()


# In[221]:


# Regresioni skor
r.score(X,Y)


# In[222]:


# Regresioni koeficijent
r.coef_


# #### *LR (modifikovana) -- 2019*

# In[223]:


# Regresija (skater)
df14= df1.loc[df1['Godina'].isin([2019])]
X = df14.iloc[:, 1].values.reshape(-1, 1)
Y = df14.iloc[:, 3].values.reshape(-1, 1)
linear_regressor = LinearRegression()
r = linear_regressor.fit(X, Y)
Y_pred = linear_regressor.predict(X)

plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()


# In[224]:


# Regresioni skor
r.score(X,Y)


# In[225]:


# Regresioni koeficijent
r.coef_


# #### *LR (modifikovana) -- 2020*

# In[226]:


# Regresija (skater)
df15= df1.loc[df1['Godina'].isin([2020])]
X = df15.iloc[:, 1].values.reshape(-1, 1)
Y = df15.iloc[:, 3].values.reshape(-1, 1)
linear_regressor = LinearRegression()
r = linear_regressor.fit(X, Y)
Y_pred = linear_regressor.predict(X)

plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()


# In[227]:


# Regresioni skor
r.score(X,Y)


# In[228]:


# Regresioni koeficijent
r.coef_


# ## *Text mining*

# #### *Tekstovi -- Ekonomija*

# In[229]:


# Učitavanje fajla (.txt)
corpus_root = 'C:\\Users\\isidora.gataric\\Anaconda3\\Lib\\site-packages\\nltk\\corpus'
ekonomija = PlaintextCorpusReader(corpus_root, 'Ekonomija.txt')
ekonom = ekonomija.words()
print(ekonom)


# In[230]:


# Koliko imamo elemenata u ovom data set-u (sirovom)
len(ekonom)


# In[231]:


# Basic plot frekvencija (pre čišćenja seta podataka)
fdist1 = nltk.FreqDist(ekonom)
fdist1.plot(40)


# In[232]:


# Dodano čišćenje seta podataka (nisam namerno izbacivala cifre)
words = ekonom
stopwords = ['com', 'www', 'generic', 'ili', '5', 'ste', 'at', '7', 'do', '2', 'O', '1', 'és', 'Ukoliko', '3', 'd', 'su', 'iz', 'and', '.', '-', 'i', 'u', 'za', 'of', 'the', 'to', 'na', 'in', 'sa', 'a', 'da', '#', 'o', 'for', 'with', 'je', 'se', 'od', 'with','is', 'will', 'koji', 'you', 'da', 'kao', 'be', 'are', 'on', 'be', 'their','pp', 'ones', 'pmp', 'bilo', 'The', 'CV', 'Vas', 'We', 'or','our','s','an', 'an', 'as', 'koja', '&', 'You', 'po', 'te', 'Za']
new_words = [word for word in words if word not in stopwords]


# In[233]:


# Najfrekventnije reči u pročišćenom setu
fdist1 = nltk.FreqDist(new_words)
print (fdist1.most_common(400))


# In[234]:


# Frekvencije prvih 30 top reči
fdist1 = nltk.FreqDist(new_words)
fdist1.plot(50)


# In[235]:


# Definisanje trigrama oko reči "nemačka"
poslovi_trigrami = nltk.Text(new_words)
poslovi_trigrami.findall(r"<.*> <.*> <godine.*> <.*> <.*>") 


# In[236]:


# Frekvencije trigrama
word_fd = nltk.FreqDist(new_words)
bigram_fd = nltk.FreqDist(nltk.trigrams(new_words))
bigram_fd.most_common()


# #### *Tekstovi -- IT*

# In[237]:


# Učitavanje fajla (.txt)
corpus_root = 'C:\\Users\\isidora.gataric\\Anaconda3\\Lib\\site-packages\\nltk\\corpus'
poslovi_all_pr = PlaintextCorpusReader(corpus_root, 'Informacione tehnologije.txt')
poslovi_all = poslovi_all_pr.words()
print(poslovi_all)


# In[238]:


# Koliko imamo elemenata u ovom data set-u (sirovom)
len(poslovi_all)


# In[239]:


# Basic plot frekvencija (pre čišćenja)
fdist1 = nltk.FreqDist(poslovi_all)
fdist1.plot(40)


# In[240]:


# Dodano čišćenje seta podataka (nisam namerno izbacivala cifre)
words = poslovi_all
stopwords = ['com', 'www', 'generic', 'ili', 'do', 'What', 'It', 'd', 'su', 'iz', 'and', '.', '-', 'i', 'u', 'za', 'of', 'the', 'to', 'na', 'in', 'sa', 'a', 'da', '#', 'o', 'for', 'with', 'je', 'se', 'od', 'with','is', 'will', 'koji', 'you', 'da', 'kao', 'be', 'are', 'on', 'be', 'their','pp', 'ones', 'pmp', 'bilo', 'The', 'CV', 'Vas', 'We', 'or','our','s','an', 'an', 'as', 'koja', 'you', 'we', 'it', 'your', 'from', 'You', 'and', 'all', '&', 'Za', 'te', 'vas','do', 'd', 'su', 'iz', 'and', '.', '-', 'i', 'u', 'za', 'of', 'the', 'to', 'na', 'in', 'sa', 'a', 'da', '#', 'o', 'for', 'with', 'je', 'se', 'od', 'with','is', 'will', 'koji', 'you', 'da', 'kao', 'be', 'are', 'on', 'be', 'their','pp', 'ones', 'pmp', 'bilo', 'The', 'CV', 'Vas', 'We', 'or','our','s','an', 'an', 'as', 'koja', 'you', 'we', 'it', 'your', 'from', 'You', 'and', 'all', '&', 'Za', 'te', 'vas', 'company', 'GDPR', 'working', 'new', 'looking', 'at', 'can', 'that', 't', '3', '2', 'up', 'Your', 'us', 'after', 'years', 'ste', 'Work']
new_words = [word for word in words if word not in stopwords]


# In[241]:


# Najfrekventnije reči u pročišćenom setu
fdist1 = nltk.FreqDist(new_words)
print (fdist1.most_common(500))


# In[242]:


# Frekvencije prvih 30 top reči
fdist1 = nltk.FreqDist(new_words)
fdist1.plot(50)


# In[243]:


# Definisanje trigrama oko reči "nemačka"
poslovi_trigrami = nltk.Text(new_words)
poslovi_trigrami.findall(r"<.*> <.*> <godine.*> <.*> <.*>") 


# In[244]:


# Frekvencije trigrama
word_fd = nltk.FreqDist(new_words)
bigram_fd = nltk.FreqDist(nltk.trigrams(new_words))
bigram_fd.most_common()


# #### *Tekstovi -- Obrazovanje*

# In[245]:


# Učitavanje fajla (.txt)
corpus_root = 'C:\\Users\\isidora.gataric\\Anaconda3\\Lib\\site-packages\\nltk\\corpus'
poslovi_all_1 = PlaintextCorpusReader(corpus_root, 'Obrazovanje.txt')
obrazovanje = poslovi_all_1.words()
print(obrazovanje)


# In[246]:


# Koliko imamo elemenata u ovom data set-u (sirovom)
len(obrazovanje)


# In[247]:


# Basic plot frekvencija (pre čiščenja)
fdist1 = nltk.FreqDist(obrazovanje)
fdist1.plot(40)


# In[248]:


# Dodano čišćenje seta podataka (nisam namerno izbacivala cifre)
words = obrazovanje
stopwords = ['com', 'www', 'generic', 'ili', '1','7','3','te','do', '5','d', '6', 'vas','su', 'iz', 'and', '.', '-', 'i', 'u', 'za', 'of', 'the', 'to', 'na', 'in', 'sa', 'a', 'da', '#', 'o', 'for', 'with', 'je', 'se', 'od', 'with','is', 'will', 'koji', 'you', 'da', 'kao', 'be', 'are', 'on', 'be', 'their','pp', 'ones', 'pmp', 'bilo', 'The', 'CV', 'Vas', 'We', 'or','our','s','an', 'an', 'as', 'koja', 'Za', 'all', 'from', 'your', 'jedan', 'GDPR']
new_words = [word for word in words if word not in stopwords]


# In[249]:


# Najfrekventnije reči u pročišćenom setu
fdist1 = nltk.FreqDist(new_words)
print (fdist1.most_common(300))


# In[250]:


# Basic plot frekvencija (pre čiščenja)
fdist1 = nltk.FreqDist(new_words)
fdist1.plot(50)


# In[251]:


# Definisanje trigrama oko reči "nemačka"
poslovi_trigrami = nltk.Text(new_words)
poslovi_trigrami.findall(r"<.*> <.*> <godine.*> <.*> <.*>") 


# In[252]:


# Frekvencije trigrama
word_fd = nltk.FreqDist(new_words)
bigram_fd = nltk.FreqDist(nltk.trigrams(new_words))
bigram_fd.most_common()


# #### *Tekstovi -- Tehničke nauke*

# In[253]:


# Učitavanje fajla (.txt)
corpus_root = 'C:\\Users\\isidora.gataric\\Anaconda3\\Lib\\site-packages\\nltk\\corpus'
poslovi_all_2 = PlaintextCorpusReader(corpus_root, 'Tehnicke nauke.txt')
tehnicke = poslovi_all_2.words()
print(tehnicke)


# In[254]:


# Koliko imamo elemenata u ovom data set-u (sirovom)
len(tehnicke)


# In[255]:


# Basic plot frekvencija (pre čišćenja)
fdist1 = nltk.FreqDist(tehnicke)
fdist1.plot(40)


# In[256]:


# Dodano čišćenje seta podataka (nisam namerno izbacivala cifre)
words = tehnicke
stopwords = ['com', 'www', 'generic', 'ili', '1', '3', 'd', '2', 'do', '5', 'su', 'iz', 'and', '.', '-', 'i', 'u', 'za', 'of', 'the', 'to', 'na', 'in', 'sa', 'a', 'da', '#', 'o', 'for', 'with', 'je', 'se', 'od', 'with','is', 'will', 'koji', 'you', 'da', 'kao', 'be', 'are', 'on', 'be', 'their','pp', 'ones', 'pmp', 'bilo', 'The', 'CV', 'Vas', 'We', 'or','our','s','an', 'an', 'as', 'koja', 'O', 'nam', 'po', 'ste', 'koje', 'GDPR']
new_words = [word for word in words if word not in stopwords]


# In[257]:


# Najfrekventnije reči u pročišćenom setu
fdist1 = nltk.FreqDist(new_words)
print (fdist1.most_common(500))


# In[258]:


# Frekvencije prvih 30 top reči
fdist1 = nltk.FreqDist(new_words)
fdist1.plot(50)


# In[259]:


# Definisanje trigrama oko reči "nemačka"
poslovi_trigrami = nltk.Text(new_words)
poslovi_trigrami.findall(r"<.*> <.*> <godine.*> <.*> <.*>") 


# In[260]:


# Frekvencije trigrama
word_fd = nltk.FreqDist(new_words)
bigram_fd = nltk.FreqDist(nltk.trigrams(new_words))
bigram_fd.most_common()


# #### *Tekstovi -- Trgovina i usluge*

# In[261]:


# Učitavanje fajla (.txt)
corpus_root = 'C:\\Users\\isidora.gataric\\Anaconda3\\Lib\\site-packages\\nltk\\corpus'
poslovi_all_3 = PlaintextCorpusReader(corpus_root, 'Trgovina i usluge.txt')
trgovina = poslovi_all_3.words()
print(trgovina)


# In[262]:


# Koliko imamo elemenata u ovom data set-u (sirovom)
len(trgovina)


# In[263]:


# Basic plot frekvencija (pre čišćenja)
fdist1 = nltk.FreqDist(trgovina)
fdist1.plot(40)


# In[264]:


# Dodano čišćenje seta podataka (nisam namerno izbacivala cifre)
words = trgovina
stopwords = ['GDPR', 'com', 'www', 'generic', '1', '2', 'će', '3', '5', 'E', 'po', 'ili', 'do', 'd', 'su', 'iz', 'and', '.', '-', 'i', 'u', 'za', 'of', 'the', 'to', 'na', 'in', 'sa', 'a', 'da', '#', 'o', 'for', 'with', 'je', 'se', 'od', 'with','is', 'will', 'koji', 'you', 'da', 'kao', 'be', 'are', 'on', 'be', 'their','pp', 'ones', 'pmp', 'bilo', 'The', 'CV', 'Vas', 'We', 'or','our','s','an', 'an', 'as', 'koja', 'Za', 'te', 'nam']
new_words = [word for word in words if word not in stopwords]


# In[265]:


# Najfrekventnije reči u pročišćenom setu
fdist1 = nltk.FreqDist(new_words)
print (fdist1.most_common(400))


# In[266]:


# Frekvencije prvih 30 top reči
fdist1 = nltk.FreqDist(new_words)
fdist1.plot(50)


# In[267]:


# Definisanje trigrama oko reči "nemačka"
poslovi_trigrami = nltk.Text(new_words)
poslovi_trigrami.findall(r"<.*> <.*> <godine.*> <.*> <.*>")


# In[268]:


# Frekvencije trigrama
word_fd = nltk.FreqDist(new_words)
bigram_fd = nltk.FreqDist(nltk.trigrams(new_words))
bigram_fd.most_common()


# #### *Tekstovi -- Turizam i ugostiteljstvo*

# In[269]:


# Učitavanje fajla (.txt)
corpus_root = 'C:\\Users\\isidora.gataric\\Anaconda3\\Lib\\site-packages\\nltk\\corpus'
poslovi_all_5 = PlaintextCorpusReader(corpus_root, 'Turizam i ugostiteljstvo.txt')
turizam = poslovi_all_5.words()
print(turizam)


# In[270]:


# Koliko imamo elemenata u ovom data set-u (sirovom)
len(turizam)


# In[271]:


# Basic plot frekvencija
fdist1 = nltk.FreqDist(turizam)
fdist1.plot(40)


# In[272]:


# Dodano čišćenje seta podataka (nisam namerno izbacivala cifre)
words = turizam
stopwords = ['GDPR', 'com', 'www', '1', '2', 'Za', 'po','3', '5', 'U', 'm','generic', 'ili', 'do', 'd', 'su', 'iz', 'and', '.', '-', 'i', 'u', 'za', 'of', 'the', 'to', 'na', 'in', 'sa', 'a', 'da', '#', 'o', 'for', 'with', 'je', 'se', 'od', 'with','is', 'will', 'koji', 'you', 'da', 'kao', 'be', 'are', 'on', 'be', 'their','pp', 'ones', 'pmp', 'bilo', 'The', 'CV', 'Vas', 'We', 'or','our','s','an', 'an', 'as', 'koja', 'te', '&', 'ž', 'će']
new_words = [word for word in words if word not in stopwords]


# In[273]:


# Najfrekventnije reči u pročišćenom setu
fdist1 = nltk.FreqDist(new_words)
print (fdist1.most_common(400))


# In[274]:


# Frekvencije prvih 30 top reči
fdist1 = nltk.FreqDist(new_words)
fdist1.plot(50)


# In[275]:


# Definisanje trigrama oko reči "nemačka"
poslovi_trigrami = nltk.Text(new_words)
poslovi_trigrami.findall(r"<.*> <.*> <godine.*> <.*> <.*>") 


# In[276]:


# Frekvencije trigrama
word_fd = nltk.FreqDist(new_words)
bigram_fd = nltk.FreqDist(nltk.trigrams(new_words))
bigram_fd.most_common()


# #### *Tekstovi -- Zdravstvo i farmacija*

# In[277]:


# Učitavanje fajla (.txt)
corpus_root = 'C:\\Users\\isidora.gataric\\Anaconda3\\Lib\\site-packages\\nltk\\corpus'
poslovi_all_6 = PlaintextCorpusReader(corpus_root, 'Zdravstvo i farmacija.txt')
zdravstvo = poslovi_all_6.words()
print(poslovi_all)


# In[278]:


# Koliko imamo elemenata u ovom data set-u (sirovom)
len(zdravstvo)


# In[279]:


# Basic plot frekvencija (pre čišćenja)
fdist1 = nltk.FreqDist(zdravstvo)
fdist1.plot(40)


# In[280]:


# Dodano čišćenje seta podataka (nisam namerno izbacivala cifre)
words = zdravstvo
stopwords = ['GDPR', 'com', 'www', 'generic', 'ili', '5', '7', 'do', 'd', 'su', 'iz', 'and', '.', '-', 'i', 'u', 'za', 'of', 'the', 'to', 'na', 'in', 'sa', 'a', 'da', '#', 'o', 'for', 'with', 'je', 'se', 'od', 'with','is', 'will', 'koji', 'you', 'da', 'kao', 'be', 'are', 'on', 'be', 'their','pp', 'ones', 'pmp', 'bilo', 'The', 'CV', 'Vas', 'We', 'or','our','s','an', 'an', 'as', 'koja','Za','te', 'vas', 'jedan', 'nam']
new_words = [word for word in words if word not in stopwords]


# In[281]:


# Najfrekventnije reči u pročišćenom setu
fdist1 = nltk.FreqDist(new_words)
print (fdist1.most_common(400))


# In[282]:


# Frekvencije prvih 30 top reči
fdist1 = nltk.FreqDist(new_words)
fdist1.plot(50)


# In[283]:


# Definisanje trigrama oko reči "nemačka"
poslovi_trigrami = nltk.Text(new_words)
poslovi_trigrami.findall(r"<.*> <.*> <godine.*> <.*> <.*>") 


# In[284]:


# Frekvencije trigrama
word_fd = nltk.FreqDist(new_words)
bigram_fd = nltk.FreqDist(nltk.trigrams(new_words))
bigram_fd.most_common()


# In[ ]:




