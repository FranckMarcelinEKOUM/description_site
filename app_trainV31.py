#!/usr/bin/env python3
"""
Script refactorisé pour la prédiction des scores avec optimisation des performances,
gestion sécurisée des connexions à la base de données, corrections de typage pour
les dates, et utilisation de méthodes vectorisées pour certaines transformations.
"""

import os
import sys
import logging
import hashlib
from datetime import datetime

import pandas as pd
import numpy as np
import joblib
from sqlalchemy import create_engine
from dotenv import load_dotenv
from prettytable import PrettyTable
import html

# Configuration du logger
logging.basicConfig(
    level=logging.DEBUG,
    filename='predict_total_score1.log',
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.info("Démarrage du script de prédiction.")

# Chargement des variables d'environnement
load_dotenv()

# ---------------------------------------------------------------------------
# Fonctions de connexion et extraction des données
# ---------------------------------------------------------------------------
def get_db_credentials():
    """Obtient les informations de connexion à la BDD depuis les variables d'environnement."""
    logger.info("Obtention des informations de connexion à la base de données.")
    db_user = os.getenv('DB_USER')
    db_password = os.getenv('DB_PASSWORD')
    db_host = os.getenv('DB_HOST', 'localhost')
    if not all([db_user, db_password, db_host]):
        logger.warning("Informations de connexion manquantes, demande en input.")
        if not db_user:
            db_user = input("Entrez votre nom d'utilisateur pour la base de données MySQL : ")
        if not db_password:
            db_password = input("Entrez votre mot de passe pour la base de données MySQL : ")
        if not db_host:
            db_host = input("Entrez l'hôte de la base de données (par défaut 'localhost') : ") or 'localhost'
    return db_user, db_password, db_host


def get_database_connection_string() -> str:
    """Construit et retourne la chaîne de connexion."""
    db_user, db_password, db_host = get_db_credentials()
    connection_string = f"mysql+mysqlconnector://{db_user}:{db_password}@{db_host}?ssl_disabled=true"
    return connection_string


def load_prediction_data(connection_string: str) -> pd.DataFrame:
    """
    Charge les données des matchs à prédire via une requête Fusionnant deux schémas.
    Utilise un context manager pour la connexion (amélioration des ressources).
    """
    try:
        engine = create_engine(connection_string, connect_args={'ssl_disabled': True})
        query = """
            SELECT *
            FROM (
                SELECT id, dataId, ch, heure, equipes, equipe1, equipe2, cotes,
                       score_equipe1, score_equipe2, total_1Mt, total_2Mt, total_score,
                       date_telechargement
                FROM essaie01.matchs
                WHERE total_score IS NULL OR score_equipe1 IS NULL OR score_equipe2 IS NULL 
                      OR total_1Mt IS NULL OR total_2Mt IS NULL
                UNION ALL
                SELECT id, dataId, ch, heure, equipes, equipe1, equipe2, cotes,
                       score_equipe1, score_equipe2, total_1Mt, total_2Mt, total_score,
                       date_telechargement
                FROM mlflow_db.matchs
                WHERE total_score IS NULL OR score_equipe1 IS NULL OR score_equipe2 IS NULL 
                      OR total_1Mt IS NULL OR total_2Mt IS NULL
            ) AS combined_data;
        """
        with engine.connect() as conn:
            data = pd.read_sql(query, conn)
        data['date_telechargement'] = pd.to_datetime(data['date_telechargement'], errors='coerce')
        logger.info("Chargement de %d lignes de données des prochains matchs.", len(data))
        print(f"[INFO] Chargement de {len(data)} lignes de données des prochains matchs.")
        return data
    except Exception as e:
        logger.error("Erreur lors du chargement des données de prédiction: %s", e)
        raise


def load_historical_data(connection_string: str) -> pd.DataFrame:
    """
    Charge les données historiques complètes via une requête UNION sur deux schémas.
    Utilise également un context manager pour la connexion.
    """
    try:
        engine = create_engine(connection_string, connect_args={'ssl_disabled': True})
        query = """
            SELECT *
            FROM (
                SELECT id, dataId, ch, heure, equipes, equipe1, equipe2, cotes,
                       score_equipe1, score_equipe2, total_1Mt, total_2Mt, total_score,
                       date_telechargement
                FROM essaie01.matchs
                WHERE total_score IS NOT NULL AND score_equipe1 IS NOT NULL 
                      AND score_equipe2 IS NOT NULL AND total_1Mt IS NOT NULL AND total_2Mt IS NOT NULL
                UNION ALL
                SELECT id, dataId, ch, heure, equipes, equipe1, equipe2, cotes,
                       score_equipe1, score_equipe2, total_1Mt, total_2Mt, total_score,
                       date_telechargement
                FROM mlflow_db.matchs
                WHERE total_score IS NOT NULL AND score_equipe1 IS NOT NULL 
                      AND score_equipe2 IS NOT NULL AND total_1Mt IS NOT NULL AND total_2Mt IS NOT NULL
            ) AS combined_data;
        """
        with engine.connect() as conn:
            historical_data = pd.read_sql(query, conn)
        historical_data['date_telechargement'] = pd.to_datetime(historical_data['date_telechargement'], errors='coerce')
        logger.info("Chargement de %d lignes historiques.", len(historical_data))
        print(f"[INFO] Chargement de {len(historical_data)} lignes historiques.")
        return historical_data
    except Exception as e:
        logger.error("Erreur lors du chargement des données historiques: %s", e)
        raise


# ---------------------------------------------------------------------------
# Fonctions de transformation et d'enrichissement
# ---------------------------------------------------------------------------
def extract_cotes(cotes: str) -> list:
    """
    Extrait les cotes depuis une chaîne du type "[1.45, 3.25, 2.75, 4.0, 5.5, 2.1]".
    Retourne une liste de 6 valeurs ou [1.001]*6 en cas d'erreur.
    """
    try:
        if not isinstance(cotes, str):
            raise ValueError("La valeur de 'cotes' n'est pas une chaîne.")
        parts = cotes.strip("[]").split(",")
        values = [float(p.strip()) for p in parts if p.strip()]
        if not values:
            raise ValueError("Aucune cote extraite.")
        values = values[:6]
        if len(values) < 6:
            values += [1.001] * (6 - len(values))
        return values
    except Exception as e:
        logger.error("Erreur dans extract_cotes pour '%s': %s", cotes, e)
        return [1.001] * 6


def safe_extract(x) -> list:
    if not isinstance(x, str):
        return [1.001] * 6
    result = extract_cotes(x)
    if not isinstance(result, (list, tuple)):
        return [1.001] * 6
    return result


def map_cotes_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Applique safe_extract sur la colonne 'cotes' et crée les colonnes de cotes."""
    df['cotes_extracted'] = df['cotes'].apply(safe_extract)
    cols = ['home_odds', 'draw_odds', 'away_odds', 'odds1', 'odds2', 'odds3']
    for i, col in enumerate(cols):
        df[col] = df['cotes_extracted'].apply(lambda x: x[i])
    return df


def add_champion_features(df: pd.DataFrame) -> pd.DataFrame:
    """Agrège les statistiques par championnat."""
    if 'cotes_extracted' not in df.columns:
        df = map_cotes_columns(df)
    df['cotes_mean'] = df['cotes_extracted'].apply(np.mean)
    champ_stats = df.groupby('ch').agg(
        champ_total_score_mean=('total_score', 'mean'),
        champ_total_score_std=('total_score', 'std'),
        champ_avg_cotes=('cotes_mean', 'mean')
    ).reset_index()
    df = df.merge(champ_stats, on='ch', how='left')
    return df


def add_meeting_features(df: pd.DataFrame) -> pd.DataFrame:
    """Agrège les statistiques par rencontre d'équipes."""
    if 'cotes_extracted' not in df.columns:
        df = map_cotes_columns(df)
    df['cotes_mean'] = df['cotes_extracted'].apply(np.mean)
    meeting_stats = df.groupby(['ch', 'equipes']).agg(
        meeting_total_score_mean=('total_score', 'mean'),
        meeting_total_score_std=('total_score', 'std'),
        meeting_avg_cotes=('cotes_mean', 'mean')
    ).reset_index()
    df = df.merge(meeting_stats, on=['ch', 'equipes'], how='left')
    return df


def add_odds_ranking(df: pd.DataFrame) -> pd.DataFrame:
    """Attribue un rang aux équipes par 'home_odds' dans chaque championnat et rencontre."""
    df['home_odds_rank'] = df.groupby(['ch', 'equipes'])['home_odds'].rank(method='min', ascending=True)
    return df


def generate_match_keys_vectorized(df: pd.DataFrame) -> pd.DataFrame:
    """
    Génère une clé unique (match_key) de façon vectorisée à partir des colonnes 'equipe1', 'equipe2' et 'cotes'.
    """
    df['equipe1_clean'] = df['equipe1'].fillna('').astype(str).str.strip().str.lower()
    df['equipe2_clean'] = df['equipe2'].fillna('').astype(str).str.strip().str.lower()
    df['cotes_clean'] = df['cotes'].fillna('').astype(str).str.strip().str.lower()
    df['cotes_hash'] = df['cotes_clean'].apply(lambda x: hashlib.md5(x.encode("utf-8")).hexdigest())
    df['match_key'] = df['equipe1_clean'] + "_" + df['equipe2_clean'] + "_" + df['cotes_hash']
    return df


def vectorized_enrich_duplicates(new_data: pd.DataFrame, historical_data: pd.DataFrame) -> pd.DataFrame:
    """
    Pour chaque nouveau match, réalise une jointure avec les historiques via "match_key"
    afin d'agréger les scores en cas de doublons et calcule pour chaque cible :
      - Le nombre de doublons trouvés.
      - La concaténation textuelle des résultats historiques.
      - La moyenne des valeurs historiques.
      - Le pourcentage de réussite en comparant la prédiction avec la moyenne historique.
      
    Si aucun doublon n’est trouvé, les colonnes correspondantes seront mises à None.
    """
    # Jointure sur "match_key" pour récupérer les données historiques associées
    merged = pd.merge(
        new_data,
        historical_data[['match_key', 'total_score', 'score_equipe1', 'score_equipe2']],
        on="match_key",
        how="inner",
        suffixes=('', '_hist')
    )
    
    if merged.empty:
        new_data['duplicate_total_score'] = None
        new_data['duplicate_score_equipe1'] = None
        new_data['duplicate_score_equipe2'] = None
        new_data['duplicate_count'] = 0
        new_data['percent_success_total'] = None
        new_data['percent_success_score_equipe1'] = None
        new_data['percent_success_score_equipe2'] = None
        return new_data

    # Agrégation : regroupe par l'id du match dans new_data
    agg = merged.groupby('id').agg(
        duplicate_count=('match_key', 'count'),
        duplicate_total_score_text=('total_score_hist', lambda x: " | ".join(map(str, x))),
        avg_total_score=('total_score_hist', 'mean'),
        duplicate_score_equipe1_text=('score_equipe1_hist', lambda x: " | ".join(map(str, x))),
        avg_score_equipe1=('score_equipe1_hist', 'mean'),
        duplicate_score_equipe2_text=('score_equipe2_hist', lambda x: " | ".join(map(str, x))),
        avg_score_equipe2=('score_equipe2_hist', 'mean')
    ).reset_index()

    # Fusionner les informations agrégées avec les données nouvelles
    new_data = pd.merge(new_data, agg, on='id', how='left')
    new_data['duplicate_count'] = new_data['duplicate_count'].fillna(0).astype(int)

    # Définition d'une fonction pour calculer le pourcentage de réussite.
    def compute_percent_success(predicted, avg_hist):
        if pd.isna(avg_hist) or avg_hist == 0:
            return None
        # Calcul à partir de la formule: 100 * (1 - |prédiction - moyenne| / moyenne)
        value = 100 * (1 - abs(predicted - avg_hist) / avg_hist)
        return max(0, round(value, 2))

    # Pour chaque cible, si la prédiction existe, on calcule le pourcentage
    for col_pred, col_avg, new_col in [
        ('predicted_total_score', 'avg_total_score', 'percent_success_total'),
        ('predicted_score_equipe1', 'avg_score_equipe1', 'percent_success_score_equipe1'),
        ('predicted_score_equipe2', 'avg_score_equipe2', 'percent_success_score_equipe2'),
    ]:
        if col_pred in new_data.columns and col_avg in new_data.columns:
            new_data[new_col] = new_data.apply(
                lambda row: compute_percent_success(row[col_pred], row[col_avg]), axis=1
            )
        else:
            new_data[new_col] = None

    # Vous pouvez aussi stocker les chaînes concaténées pour un affichage éventuel
    new_data['duplicate_total_score'] = new_data['duplicate_total_score_text']
    new_data['duplicate_score_equipe1'] = new_data['duplicate_score_equipe1_text']
    new_data['duplicate_score_equipe2'] = new_data['duplicate_score_equipe2_text']

    return new_data


def preprocess_data(data: pd.DataFrame, preprocessor, historical_data: pd.DataFrame):
    """
    Prétraite les données pour la prédiction en fusionnant les historiques et les données
    nouvelles, normalise et enrichit les caractéristiques, puis transforme via le préprocesseur.
    """
    logger.info("Prétraitement des données pour la prédiction.")
    print("[INFO] Début du prétraitement des données pour la prédiction.")
    try:
        # Concatène les données historiques et nouvelles
        combined_data = pd.concat([historical_data, data], ignore_index=True)
        # Normaliser les dates pour éviter les problèmes de comparaison de type
        combined_data['date_telechargement'] = pd.to_datetime(combined_data['date_telechargement'], errors='coerce').dt.normalize()
        combined_data = combined_data.sort_values('date_telechargement')
        print("[INFO] Fusion des données historiques et nouvelles réalisée.")

        logger.info("Conversion de 'heure' et 'date_telechargement'.")
        combined_data['heure'] = pd.to_datetime(combined_data['heure'], format='%H:%M', errors='coerce')
        combined_data = combined_data.dropna(subset=['heure', 'cotes', 'equipe1', 'equipe2', 'date_telechargement']).copy()

        logger.info("Extraction des cotes et création des colonnes dédiées.")
        combined_data = map_cotes_columns(combined_data)

        logger.info("Création des caractéristiques temporelles.")
        combined_data['heure_minutes'] = combined_data['heure'].dt.hour * 60 + combined_data['heure'].dt.minute
        combined_data['day_of_week'] = combined_data['heure'].dt.dayofweek

        # Enrichissement hiérarchique identique au training
        combined_data = add_champion_features(combined_data)
        combined_data = add_meeting_features(combined_data)
        combined_data = add_odds_ranking(combined_data)
        print("[INFO] Enrichissement hiérarchique appliqué.")

        new_data = combined_data[combined_data['id'].isin(data['id'])].copy()
        new_data = generate_match_keys_vectorized(new_data)

        features = [
            'equipe1', 'equipe2', 'home_odds', 'draw_odds', 'away_odds',
            'odds1', 'odds2', 'odds3', 'heure_minutes', 'day_of_week',
            'champ_total_score_mean', 'champ_total_score_std', 'champ_avg_cotes',
            'meeting_total_score_mean', 'meeting_total_score_std', 'meeting_avg_cotes',
            'home_odds_rank'
        ]
        X = new_data[features]
        logger.info("Transformation des données avec le préprocesseur chargé.")
        X_preprocessed = preprocessor.transform(X)
        logger.info("Prétraitement terminé avec succès.")
        print("[INFO] Prétraitement terminé.")
        return X_preprocessed, new_data
    except Exception as e:
        logger.error("Erreur dans preprocess_data: %s", e)
        raise


def generate_html_table(data: pd.DataFrame) -> str:
    """Génère un tableau HTML affichant les prédictions enrichies."""
    logger.info("Génération du tableau HTML avec les prédictions enrichies.")
    try:
        table = PrettyTable()
        table.field_names = [
            "ID", "Date", "Championnat", "Équipe 1", "Équipe 2", "Heure",
            "Prédiction Total Score", "Pourc. Réussite Total", "Résultat Total Doublon",
            "Prédiction Score Équipe 1", "Pourc. Réussite Équipe 1", "Résultat Équipe 1 Doublon",
            "Prédiction Score Équipe 2", "Pourc. Réussite Équipe 2", "Résultat Équipe 2 Doublon",
            "Doublon Trouvé"
        ]

        def cell_content(value):
            return "N/A" if value is None else f"<div style='max-width:150px; overflow-x:auto; white-space: nowrap;'>{value}</div>"

        for _, row in data.iterrows():
            heure_str = row['heure'].strftime('%H:%M') if pd.notnull(row['heure']) else ''
            table.add_row([
                row['id'],
                row['date_telechargement'],
                row['ch'],
                row['equipe1'],
                row['equipe2'],
                heure_str,
                f"{row['predicted_total_score']:.2f}",
                f"{row.get('percent_success_total', 'N/A'):.2f}%" if row.get('percent_success_total') is not None else "N/A",
                cell_content(row.get('duplicate_total_score')),
                f"{row['predicted_score_equipe1']:.2f}",
                f"{row.get('percent_success_score_equipe1', 'N/A'):.2f}%" if row.get('percent_success_score_equipe1') is not None else "N/A",
                cell_content(row.get('duplicate_score_equipe1')),
                f"{row['predicted_score_equipe2']:.2f}",
                f"{row.get('percent_success_score_equipe2', 'N/A'):.2f}%" if row.get('percent_success_score_equipe2') is not None else "N/A",
                cell_content(row.get('duplicate_score_equipe2')),
                "Oui" if row.get('duplicate_count', 0) >= 1 else "Non"
            ])
        html_table = table.get_html_string(attributes={"border": "1", "style": "margin: 0 auto;"})
        html_table = html.unescape(html_table)
        html_content = f"""
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                body {{
                    background-color: #000;
                    color: #fff;
                    font-family: Arial, sans-serif;
                }}
                .table-container {{
                    overflow-x: auto;
                    max-height: 80vh;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                }}
                th, td {{
                    padding: 10px;
                    border: 1px solid #fff;
                    text-align: left;
                    white-space: nowrap;
                }}
                th {{
                    background-color: #333;
                    position: sticky;
                    top: 0;
                    z-index: 2;
                }}
                tr:nth-child(even) {{
                    background-color: #222;
                }}
            </style>
            <title>Prédictions des Scores</title>
        </head>
        <body>
            <h1 style="text-align: center;">Prédictions des Scores</h1>
            <div class="table-container">
                {html_table}
            </div>
        </body>
        </html>
        """
        logger.info("Tableau HTML généré avec succès.")
        return html_content
    except Exception as e:
        logger.error("Erreur lors de la génération du tableau HTML: %s", e)
        raise


def load_model_artifacts() -> tuple:
    """
    Charge le modèle et le préprocesseur depuis le dossier des artefacts.
    Retourne (best_model, preprocessor).
    """
    model_artifacts_dir = 'model_artifacts3'
    if not os.path.exists(model_artifacts_dir):
        raise FileNotFoundError(f"Le répertoire {model_artifacts_dir} n'existe pas.")
    files = os.listdir(model_artifacts_dir)
    preprocessor_files = sorted([f for f in files if f.endswith("preprocessor_multioutput.pkl")], reverse=True)
    model_files = sorted([f for f in files if f.endswith("_multioutput_model.pkl")], reverse=True)
    if not model_files or not preprocessor_files:
        raise FileNotFoundError("Aucun fichier modèle ou préprocesseur trouvé dans le répertoire des artefacts.")
    model_path = os.path.join(model_artifacts_dir, model_files[0])
    preprocessor_path = os.path.join(model_artifacts_dir, preprocessor_files[0])
    best_model_name = model_files[0].split('_multioutput_model')[0]
    best_model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    logger.info("Modèle '%s' et préprocesseur chargés avec succès.", best_model_name)
    print(f"[INFO] Modèle '{best_model_name}' et préprocesseur chargés avec succès.")
    return best_model, preprocessor


# ---------------------------------------------------------------------------
# Point d'entrée principal du script
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        print("[INFO] Début de l'exécution de app_trainV31.py")
        logger.info("Démarrage de l'exécution principale du script.")
        connection_string = get_database_connection_string()
        print("[INFO] Chaîne de connexion récupérée.")

        prediction_data = load_prediction_data(connection_string)
        historical_data = load_historical_data(connection_string)

        # Générer la clé de match pour les données historiques de manière vectorisée
        historical_data = generate_match_keys_vectorized(historical_data)

        # Normaliser les dates et filtrer les enregistrements de la journée
        today = pd.Timestamp.today().normalize()
        historical_data['date_telechargement'] = pd.to_datetime(historical_data['date_telechargement'], errors='coerce').dt.normalize()
        historical_today = historical_data[historical_data['date_telechargement'] >= today].copy()
        print(f"[INFO] {len(historical_today)} enregistrements historiques trouvés pour aujourd'hui.")
        logger.info("Enregistrements historiques trouvés pour aujourd'hui : %d", len(historical_today))

        # Chargement du modèle et du préprocesseur
        best_model, preprocessor = load_model_artifacts()

        X_preprocessed, data_processed = preprocess_data(prediction_data, preprocessor, historical_data)
        print(f"[INFO] Prétraitement terminé. {len(data_processed)} matchs détectés pour les prédictions.")
        logger.info("Prétraitement terminé. %d matchs détectés.", len(data_processed))

        if X_preprocessed.shape[0] == 0:
            logger.warning("Aucune donnée disponible pour la prédiction.")
            print("Aucune donnée disponible pour la prédiction.")
        else:
            print("[INFO] Réalisation des prédictions en cours...")
            logger.info("Réalisation des prédictions.")
            y_pred = best_model.predict(X_preprocessed)
            # Extraction des prédictions dans le DataFrame
            data_processed['predicted_total_score'] = y_pred[:, 0]
            data_processed['predicted_score_equipe1'] = y_pred[:, 1]
            data_processed['predicted_score_equipe2'] = y_pred[:, 2]

            # Enrichissement des doublons via clé de match
            data_processed = vectorized_enrich_duplicates(data_processed, historical_today)
            data_processed = data_processed[data_processed['duplicate_count'] >= 1]
            print(f"[INFO] Nombre de matchs ayant des doublons aujourd'hui : {len(data_processed)}")
            logger.info("Nombre de matchs ayant des doublons aujourd'hui : %d", len(data_processed))

            print("[INFO] Génération du fichier HTML...")
            html_output = generate_html_table(data_processed)
            with open('predictions.html', 'w', encoding='utf-8') as f:
                f.write(html_output)
            print("[INFO] Fichier HTML 'predictions.html' généré avec succès.")
            logger.info("Fichier HTML 'predictions.html' généré avec succès.")
            print("Les prédictions ont été réalisées et le fichier 'predictions.html' a été créé, affichant uniquement les matchs avec doublons du jour.")

        print("[INFO] Script exécuté avec succès.")
        logger.info("Script exécuté avec succès.")
    except Exception as e:
        logger.error("Erreur lors de l'exécution du script : %s", e)
        print(f"Une erreur s'est produite : {e}")
        sys.exit(1)

