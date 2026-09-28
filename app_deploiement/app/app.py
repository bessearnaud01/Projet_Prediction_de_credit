from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, jsonify, render_template

# app.py est dans app_deploiement/app/ -> on remonte à app_deploiement/
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_FILE = BASE_DIR / 'model.pkl'
DATA_FILE = BASE_DIR / 'data' / 'train_u6lujuX_CVtuZ9i.csv'

# Les 3 variables retenues (même ordre partout : entraînement, formulaire, API)
FEATURES = ['Credit_History', 'Married', 'CoapplicantIncome']
TARGET = 'Loan_Status'

app = Flask(__name__,
            template_folder=str(BASE_DIR / 'templates'),
            static_folder=str(BASE_DIR / 'static'))


def preparer_X(df):
    """Nettoie et encode les 3 variables explicatives."""
    X = df[FEATURES].copy()
    # Married : Yes/No -> 1/0 (valeurs manquantes -> valeur la plus fréquente)
    X['Married'] = X['Married'].map({'Yes': 1, 'No': 0, 1: 1, 0: 0})
    X['Married'] = X['Married'].fillna(X['Married'].mode()[0])
    # Credit_History : 1.0 / 0.0, manquants -> valeur la plus fréquente
    X['Credit_History'] = X['Credit_History'].fillna(X['Credit_History'].mode()[0])
    # CoapplicantIncome : numérique, manquants -> médiane
    X['CoapplicantIncome'] = pd.to_numeric(X['CoapplicantIncome'], errors='coerce')
    X['CoapplicantIncome'] = X['CoapplicantIncome'].fillna(X['CoapplicantIncome'].median())
    return X.astype(float)


def entrainer_modele():
    """Entraîne une régression logistique (classification Y/N du prêt)."""
    dataset = pd.read_csv(DATA_FILE)
    X_2 = preparer_X(dataset)
    y = dataset[TARGET].map({'Y': 1, 'N': 0})

    # Mise à l'échelle : sinon CoapplicantIncome (milliers) écrase les variables 0/1
    classifier = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    classifier.fit(X_2.values, y)
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(classifier, f)
    return classifier


def charger_modele():
    """Charge model.pkl. S'il est absent, trop ancien ou illisible, réentraîne."""
    try:
        with open(MODEL_FILE, 'rb') as f:
            m = pickle.load(f)
        if getattr(m, 'n_features_in_', None) != len(FEATURES):
            raise ValueError("ancien modèle (mauvais nombre de variables)")
        return m
    except Exception as e:
        print(f"model.pkl inutilisable ({e}) -> réentraînement du modèle")
        return entrainer_modele()


model = charger_modele()


def lire_features(source):
    """Lit les 3 valeurs par leur nom (et non par l'ordre du formulaire)."""
    married = str(source['Married']).strip().lower()
    married = 1.0 if married in ('1', 'yes', 'oui', 'true') else 0.0
    return np.array([[float(source['Credit_History']),
                      married,
                      float(source['CoapplicantIncome'])]])


def predire(features):
    classe = int(model.predict(features)[0])
    proba = float(model.predict_proba(features)[0][1])
    return classe, proba


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    '''Résultat affiché dans l'interface HTML'''
    try:
        features = lire_features(request.form)
    except (KeyError, ValueError):
        return render_template('index.html',
                               prediction_text='Merci de remplir les 3 champs correctement.')
    classe, proba = predire(features)
    texte = 'Prêt accordé' if classe == 1 else 'Prêt refusé'
    return render_template('index.html',
                           prediction_text=f'{texte} (probabilité d\'acceptation : {proba:.0%})')


@app.route('/predict_api', methods=['POST'])
def predict_api():
    '''
    Appel direct, ex. :
    {"Credit_History": 1, "Married": "Yes", "CoapplicantIncome": 1500}
    '''
    data = request.get_json(force=True)
    try:
        features = lire_features(data)
    except (KeyError, ValueError) as e:
        return jsonify({'erreur': f'donnée manquante ou invalide : {e}'}), 400
    classe, proba = predire(features)
    return jsonify({'Loan_Status': 'Y' if classe == 1 else 'N',
                    'probabilite': round(proba, 3)})


if __name__ == "__main__":
    app.run(debug=True)