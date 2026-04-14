import os 
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, classification_report, accuracy_score

#citim fisierul cu datele 
base_path = os.path.dirname(__file__)
csv_read = os.path.join(base_path, 'StudentsPerformance.csv')
data = pd.read_csv(csv_read)

output_file = open("date.txt","w",encoding ="utf-8")
output_file.write(f"Afișăm datele înainte de prelucrare: \n {data.head()}\n\n")

#creem un folder pentru toate figurile din lucrare
output_dir = 'Grafice_Noi'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

#analiza brută a datelor 
note = ['math score', 'reading score', 'writing score']

for nota in note:
    plt.figure(figsize=(8, 5))
    sns.histplot(data[nota], bins=20, kde=True, color='teal')
    plt.title(f'Distribuția Scorurilor: {nota.capitalize()}')
    plt.xlabel('Punctaj (0-100)')
    plt.ylabel('Frecvență')
    plt.savefig(os.path.join(output_dir, f'distributie_{nota.replace(" ", "_")}.png'))
    plt.close()
    
plt.figure(figsize=(8, 6))
correlation_matrix = data[note].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='YlGnBu', fmt='.2f')
plt.title('Corelația între cele 3 materii')
plt.savefig(os.path.join(output_dir, 'corelatie_note.png'))
plt.close()  

sns.lmplot(x='reading score', y='writing score', data = data,
           height=6, aspect =1.3, 
           scatter_kws={'alpha':0.4}, line_kws={'color':'red','linewidth':2})

plt.title('Relația Liniară cu Linie de Regresie: Reading vs Writing')
plt.xlabel('reading score')
plt.ylabel('writing score')

plt.savefig(os.path.join(output_dir,'relatie_liniara.png'))

print(f'\n Graficele se găsesc în: "{output_dir}" ')

#regresii 
data_reg = data.copy()
data_reg = pd.get_dummies(data_reg, columns =['gender','lunch','test preparation course'],drop_first=True)

education_order = {'some high school': 0, 'high school': 1, 'some college':2,
                   "associate's degree":3, "bachelor's degree":4, "master's degree":5}

data_reg['parental level of education'] = data_reg['parental level of education'].map(education_order)

X = data_reg[['parental level of education','gender_male','lunch_standard','test preparation course_none']]

predictii_test ={}
valori_reale ={}

output_file.write("Rezultate regresii individuale: \n\n")

for materie in note: 
    y = data_reg[materie]
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=42)

    model =RandomForestRegressor(n_estimators=100,random_state=42)
    model.fit(X_train,y_train)

    y_pred = model.predict(X_test)

    predictii_test[materie] = y_pred
    valori_reale[materie] = y_test

    #evaluare
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test,y_pred)
    output_file.write(f"Materia: {materie}\n R2 Score: {r2:.3f}\n\n Eroare medie:{mae: .3f} puncte \n\n")

medie_predictii =(predictii_test['math score'] +predictii_test['writing score'] + predictii_test['reading score']) /3
medie_reala =(valori_reale['math score'] + valori_reale['writing score']+valori_reale['reading score'])/3

def transforma_in_clasa(scor):
    if scor <60: return "Low"
    elif scor<80: return "Medium"
    else: return "High"

y_clasa_prezisa = [transforma_in_clasa(s) for s in medie_predictii]
y_clasa_reala = [transforma_in_clasa(s) for s in medie_reala]

output_file.write("Performanta prin medie \n")
acc = accuracy_score(y_clasa_reala,y_clasa_prezisa)
output_file.write(f"Acuratețe finală (după regresie): {acc:.3f}\n\n")
output_file.write("Raport detaliat: \n")
output_file.write(classification_report(y_clasa_reala,y_clasa_prezisa))

output_file.close()
print(f"Proces finalizat! Rezultatele sunt în 'date.txt' și graficele în '{output_dir}'.")