import os 
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#citim fisierul cu datele 
base_path = os.path.dirname(__file__)
csv_read = os.path.join(base_path, 'StudentsPerformance.csv')
data = pd.read_csv(csv_read)

output_file = open("output.txt","w",encoding ="utf-8")
output_file.write(f"Afișăm datele înainte de prelucrare: \n {data.head()}\n\n")

#creem un folder pentru toate figurile din lucrare
output_dir = 'Grafice_Licenta'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

#analiza brută a datelor 
coloane = ['gender', 'lunch', 'test preparation course', 'parental level of education']

for col in coloane:
    plt.figure(figsize=(8, 5))
    order = data[col].value_counts().index
    sns.countplot(x=col, data=data, palette='viridis', order=order)
    
    total = len(data[col])
    for p in plt.gca().patches:
        percentage = '{:.1f}%'.format(100 * p.get_height()/total)
        plt.gca().annotate(percentage, (p.get_x() + p.get_width()/2., p.get_height()), 
                           ha='center', va='baseline', fontsize=12, color='black', xytext=(0, 5),
                           textcoords='offset points')
                           
    plt.title(f'Distribuția Procentuală: {col}')
    file_path = os.path.join(output_dir, f'distributie_{col.replace("/","_").replace(" ","_")}.png')
    plt.savefig(file_path)
    plt.close()
    plt.show() 

note = ['math score', 'reading score', 'writing score']

for nota in note:
    plt.figure(figsize=(8, 5))
    sns.histplot(data[nota], bins=10, kde=True, color='skyblue') # kde adaugă acea linie curbă de distribuție
    plt.title(f'Distribuția Notelor la {nota.capitalize()}')
    plt.xlabel('Punctaj')
    plt.ylabel('Număr Studenți')
    file_path = os.path.join(output_dir, f'histograma_{nota.replace(" ", "_")}.png')
    plt.savefig(file_path)
    plt.close() 
    
#I Analiza datelor 
#definim coloane noi pentru performanță
data['Total_Score'] = (data['math score'] + data['reading score'] + data['writing score']) / 3

# 2. Creăm coloana de performanță (Class)
def categorisire(scor):
    if scor < 60: return 'Low'
    elif scor < 80: return 'Medium'
    else: return 'High'

data['Performance_Class'] = data['Total_Score'].apply(categorisire)

output_file.write(f"Afișăm datele după creaarea noii coloane de performanță: \n {data.head()}\n\n" )

# Folosim coloana Performance_Class creată anterior
plt.figure(figsize=(8, 5))
order = ['Low', 'Medium', 'High']
sns.countplot(x='Performance_Class', data=data, order=order, palette='magma')

# Adăugăm procentele deasupra barelor
total = len(data)
for p in plt.gca().patches:
    percentage = '{:.1f}%'.format(100 * p.get_height()/total)
    plt.gca().annotate(percentage, (p.get_x() + p.get_width()/2., p.get_height()), 
                       ha='center', va='baseline', fontsize=12, xytext=(0, 5),
                       textcoords='offset points')

plt.title('Distribuția Performanței Academice (Medie)')
plt.savefig(os.path.join(output_dir, 'performanta_generala.png'))
plt.close()

#Label encoding for binary category
data_enc = data.copy() #copie fisier
le = LabelEncoder()

data_enc['gender'] = le.fit_transform(data['gender']) #female = 0, male = 1
data_enc['lunch'] = le.fit_transform(data['lunch']) #standard =1, free/reduced = 0
data_enc['test preparation course'] = le.fit_transform(data['test preparation course']) #none = 1, #completed = 0

education_order = { 

    'some high school': 0, 'high school':1,
    'some college': 2, "associate's degree":3,
    "bachelor's degree": 4, "master's degree":5  
}

data_enc['parental level of education'] = data_enc['parental level of education'].map(education_order)
data_enc['Performance_Class'] = data_enc['Performance_Class'].map({'Low':0,'Medium':1,'High':2})


output_file.write(f"Afișăsm datele după label encoding: \n {data_enc.head()} \n\n")

plt.figure(figsize=(10, 8))
correlation_matrix = data_enc.select_dtypes(include=[np.number]).corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.savefig(os.path.join(output_dir, 'Heatmap_Corelatie.png'))
plt.close()
print(f'\n Graficele se găsesc în: "{output_dir}" ')

#pregătim datele pentru modalare
X = data_enc[['gender','lunch','test preparation course','parental level of education']]
Y = data_enc['Performance_Class']

X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.2, random_state=42)
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X_train, Y_train)

#evaluare & predictie
y_pred = model_rf.predict(X_test)

if os.path.isfile("output.txt"):

    output_file.write(f"Acuratetea modelului: \n{ accuracy_score(Y_test, y_pred)}\n\n")
    output_file.write(f"\nRaport de clasificare: \n{classification_report(Y_test,y_pred)} \n\n")
    output_file.close()
    print("Operație realizată cu succes!")
else: print("Fișiserul nu a putut fi deschis. ")

#profilele studentilor 
#X_cluster = data_enc[['lunch','test preparation course', 'parental level of education','Total_Score']]
#kmeans = 