**Prédiction du Churn Client (désabonnement)**
***Développer un pipeline complet de Machine Learning supervisé pour prédire le désabonnement des clients dans une entreprise de télécommunications.*** 

**Lien trello: https://trello.com/invite/b/68e39c24c65c2af01440ea36/ATTId82e20b9238e272717fa80f6a5d6bc66C26F1B54/brief1**

**🧩 Étape 1 : Analyse exploratoire des données (EDA)**
***Le dataset contient 7043 lignes et 21 colonnes :float64 (2), int64 (2), object (17)***

***🔍 Description et analyse initiale:***
***Utiliser describe() et info() pour examiner les types de données et détecter les anomalies.***
***Convertir la colonne TotalCharges (type string) en float.***
***Identifier et supprimer les doublons.***
***Traiter les valeurs manquantes (remplacement par la moyenne).***

***📊 Visualisations:***
***Analyse univariée : comparer le nombre de clients désabonnés vs abonnés.***
***Analyse bivariée : utiliser un countplot pour visualiser la relation entre Contract et Churn.***

***🔗 Corrélation:***
***Générer une matrice de corrélation pour observer les relations entre les variables.***


**🧩 Étape 2 : Préparation des données (pipeline.py)**
***Encodage des variables catégorielles avec LabelEncoder.***

***Splitter les colonnes pour train et test : X (toutes les colonnes sauf Churn et customerID) et y (Churn) avec train_test_split.***

***Utiliser MinMaxScaler() pour réduire la variété des valeurs en utilisant fit_transform et VarianceThreshold() pour supprimer les colonnes qui ne varient pas.***

***Entraîner les trois modèles (LogisticRegression, SVC, RandomForestClassifier) afin d’identifier quel modèle est meilleur que l’autre.***

***Identifier les scores de chaque modèle (accuracy_score, classification_report, precision_score, recall_score, f1_score) et, puisque les données ne sont pas équilibrées, se baser sur (f1_score et precision_score) tout en traçant la courbe roc_auc_score et la confusion_matrix.***

**🧩 Étape 3 : Tests unitaires avec Pytest (test_pipeline.py)**
***Dimensions cohérentes entre X et y après le split.***

**🧩 Étape 4 : Rédiger le rapport (README.md)**
***Tracer un tableau récapitulatif qui décrit la différence entre les trois modèles au niveau des scores, la courbe ROC et la matrice de confusion.***





**Tableau récapitulatif des performances des modèles :**

|Model                 |accuracy|precision_score |recall_score|f1_score|roc_auc_score  |ROC         |matrice confusion  |
|----------------------|--------|----------------|------------|--------|---------------|------------|---------------|
|**SVC**               |0.816   |0.68            |0.57        |0.62    |0.60           |<img src="SVC/courbe_roc.png" width="120"/>|<img src="SVC/matrice_confusion.png" width="120"/>       |
|----------------------|--------|----------------|------------|--------|---------------|------------|---------------|
|**Random forest**     |0.79    |0.65            |0.45        |0.53    |0.71           |<img src="RandomForest/curv_roc.png" width="120"/>|<img src="RandomForest/matrice_confusion.png" width="120"/> 
|----------------------|--------|----------------|------------|--------|---------------|------------|---------------|
|**LogisticRegression**|0.81    |0.67            |0.57        |0.62    |0.5            |<img src="logisticregression/Figure_1.png" width="120"/>|<img src="logisticregression/matrice_confusion.png" width="120"/> 
|----------------------|--------|----------------|------------|--------|---------------|------------|---------------|

**SVC est le meilleur modèle globalement car Il a la meilleure Accuracy (0.816) et un bon F1-score (0.62)**

***comparaison du performance des trois models:***
![comparaison du performance ](bar.png)
