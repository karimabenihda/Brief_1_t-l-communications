import matplotlib.pyplot as plt
import pandas as pd

# Données du tableau
data = {
    'Modèle': ['SVC', 'Random Forest', 'Logistic Regression'],
    'Accuracy': [0.816, 0.79, 0.81],
    'Precision': [0.68, 0.65, 0.67],
    'Recall': [0.57, 0.45, 0.57],
    'F1-score': [0.62, 0.53, 0.62],
    'ROC AUC': [0.60, 0.71, 0.50]
}

df = pd.DataFrame(data)

# Afficher un barplot
df.plot(x='Modèle', kind='bar', figsize=(9, 5))
plt.title("Comparaison des performances des modèles")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.legend(title="Métriques")
plt.show()
