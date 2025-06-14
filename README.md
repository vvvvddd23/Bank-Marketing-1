Acest proiect se concentrează pe clasificarea clienților unei bănci după probabilitatea de a subscrie un depozit pe termen lung, o problemă esențială pentru optimizarea marketingului bancar și creșterea profitabilității. Scopul este de a demonstra utilitatea învățării automate în sectorul financiar și compararea performanței unor modele de clasificare (Regresie Logistică, Arbori Decizionali, Random Forest) pentru a identifica cea mai eficientă metodă.  
Setul de date include informații demografice (vârstă, ocupație, educație) și comportamentale (sold cont, istoric contacte), preprocesate pentru analiză.Proiectul a fost integrat în cursuri precum „Bazele Învățării Automate” (preprocesare și clasificare), „Vizualizarea Datelor” (analize statistice) și „Aplicații Practice” (interfață Streamlit pentru predicții), oferind o perspectivă practică asupra aplicării învățării automate în domeniul bancar.
Set de date și caracteristici 
Setul de date „Bank Marketing” (UCI Repository, Moro et al., 2014) conține 45.211 exemple: 11,7% pozitive (subscrieri) și 88,3% negative. Este împărțit 80% antrenament și 20% test, cu eșantionare stratificată. 
Preprocesare:
Există valori lipsă. Outlierii numerici sunt eliminați prin IQR.
Variabilele categorice (ex. „job”, „education”) sunt codificate prin one-hot encoding. Variabilele numerice sunt standardizate cu StandardScaler (medie 0, deviație 1).
 
![image](https://github.com/user-attachments/assets/8c6e64fe-b739-4897-afa1-9844b28ac974)
Metode 
Regresia Logistică este un algoritm de clasificare liniară care estimează probabilitatea unui eveniment folosind o funcție logistică
•	Algoritmul este simplu, rapid și ușor de interpretat, dar presupune o relație liniară între caracteristici și variabila țintă.
Arbori Decizionali: Împart datele prin reguli bazate pe caracteristici, optimizând impuritatea. Asigură interpretabili, gestionează relații neliniare, dar are riscul de overfitting.
Random Forest: Ansamblu de arbori decizionali, antrenat pe subseturi aleatoare, reducând overfitting-ul și crescând acuratețea, deși mai complex.
Matrice de confuzie: Pentru vizualizarea distribuției predicțiilor corecte și incorecte.
- Regresia Logistică a avut cea mai mică acuratețe și recall, ceea ce indică dificultăți în captarea relațiilor neliniare din date.
![image](https://github.com/user-attachments/assets/c850b4a2-b774-4b68-85d4-0f7995cc2690)

- Arborele Decizional a oferit o îmbunătățire semnificativă a recall-ului și F1-Score, dar este predispus la overfitting.
  ![image](https://github.com/user-attachments/assets/424de348-7039-48d1-8912-7f601cd555ee)
  
- Random Forest a obținut cele mai bune rezultate, datorită capacității de a generaliza și de a gestiona datele neechilibrate.
  ![image](https://github.com/user-attachments/assets/cca2b78a-cd56-4274-a509-4b5a090e920b)
  Acest proiect a demonstrat utilitatea învățării automate în sectorul bancar, comparând trei modele de clasificare pentru predicția subscrierii de depozite pe termen lung. Regresia Logistică a fost rapidă și ușor de interpretat, dar cu acuratețe și recall scăzute. Arborii Decizionali au îmbunătățit recall-ul și F1-Score, dar au fost predispuși la overfitting. Random Forest a obținut cele mai bune rezultate, datorită capacității de generalizare și gestionare a datelor neechilibrate. Alegerea modelului depinde de priorități: Regresia Logistică pentru viteză, Arborii Decizionali pentru interpretabilitate, iar Random Forest pentru performanță maximă. Proiectul subliniază importanța selecției atente a algoritmilor și hiperparametrilor în funcție de cerințele specifice.
  
