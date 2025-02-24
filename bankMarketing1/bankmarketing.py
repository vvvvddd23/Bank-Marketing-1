# Importăm librăriile necesare
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import streamlit as st

# 1. Încărcarea setului de date
file_path = r'D:\pyton\laborator2\bankMarketing1\bank\bank-full.csv'
data = pd.read_csv(file_path, sep=';')

# 2. Preprocesarea datelor
# a) Vizualizarea datelor
st.title('Vizualizare Date')
st.write('Primele 5 rânduri ale setului de date:')
st.write(data.head())

# b) Statistici descriptive
st.subheader('Statistici descriptive')
st.write(data.describe())

# c) Tratarea valorilor lipsă
st.subheader('Valori lipsă')
st.write(data.isnull().sum())  # Verificăm valorile lipsă
# Eliminăm valorile lipsă, dacă există
data.dropna(inplace=True)

# d) Detectarea și eliminarea outlierilor (folosind IQR)
for col in data.select_dtypes(include=[np.number]).columns:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    data = data[~((data[col] < (Q1 - 1.5 * IQR)) | (data[col] > (Q3 + 1.5 * IQR)))]

# 3. Elaborarea modelelor de clasificare
# Transformăm variabila țintă în valori binare
data['y'] = data['y'].map({'yes': 1, 'no': 0})

# Pregătirea datelor
X = data.drop('y', axis=1)
y = data['y']
X = pd.get_dummies(X, drop_first=True)  # One-hot encoding pentru variabilele categorice

# Împărțirea setului de date
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizarea datelor
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# a) Modelul de regresie logistică
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_scaled, y_train)
log_preds = log_model.predict(X_test_scaled)

# b) Modelul arborelui decizional
tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)
tree_preds = tree_model.predict(X_test)

# c) Modelul Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

# 4. Compararea performanțelor modelelor
st.subheader('Compararea modelelor')
models = {
    'Regresie Logistică': (log_preds, log_model),
    'Arbore Decizional': (tree_preds, tree_model),
    'Random Forest': (rf_preds, rf_model)
}

for name, (preds, model) in models.items():
    st.write(f'**{name}**')
    st.write('Acuratețe:', accuracy_score(y_test, preds))
    st.write('Raport de clasificare:')
    st.text(classification_report(y_test, preds))
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Matrice de Confuzie - {name}')
    plt.xlabel('Prevăzut')
    plt.ylabel('Real')
    st.pyplot(plt.gcf())
    plt.clf()

# 5. Interfață simplă cu Streamlit pentru predicții noi
st.title('Predicții noi')
user_input = {}
for col in X.columns:
    user_input[col] = st.text_input(f'Introduceți valoarea pentru {col}', value='0')

if st.button('Realizează predicția'):
    try:
        user_df = pd.DataFrame([user_input])
        user_df = pd.get_dummies(user_df, drop_first=True)
        user_df = user_df.reindex(columns=X.columns, fill_value=0)
        user_scaled = scaler.transform(user_df)
        prediction = rf_model.predict(user_scaled)
        st.write('Rezultatul predicției:', 'Yes' if prediction[0] == 1 else 'No')
    except ValueError as e:
        st.error(f'Valori invalide introduse: {e}')
