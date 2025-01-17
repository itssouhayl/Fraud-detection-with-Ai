# Fraud-detection-with-Ai

dataset link: https://www.kaggle.com/datasets/kartik2112/fraud-detection

Un fichier pour la sauvegarde des logs (detected_frauds.csv) doit être créé dans le même répertoire que le script.

Il est recommandé d'ajouter le chemin complet des fichiers dans le fichier "app.py".

L'interface que nous avons créée permet de détecter automatiquement les transactions frauduleuses dans un jeu de données en temps réel. Lorsqu'une nouvelle transaction est identifiée comme frauduleuse, elle est signalée et enregistrée dans un fichier de log. L'application est conçue pour surveiller en continu les nouvelles transactions ajoutées au jeu de données, garantissant ainsi une détection rapide et efficace des fraudes.

Par ailleurs, le modèle entraîné a été sauvegardé sous le nom de "model.pkl" pour être utilisé dans l'application. Il est impératif d'utiliser la même dataset ou une dataset possédant les mêmes champs que la dataset originale pour garantir le bon fonctionnement de l'application et la cohérence des prédictions.
