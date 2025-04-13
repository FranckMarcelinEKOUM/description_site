import os
from dotenv import load_dotenv
import requests
import json
import mysql.connector
import logging
from datetime import datetime

# Charger les variables d'environnement et configurer le logging
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration de la base de données
db_config = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD'),
    'database': os.getenv('DB_DATABASE', 'sport_scores')
}

# Fichier de checkpoint pour mémoriser la dernière date traitée
CHECKPOINT_FILE = "last_checkpoint.txt"
DEFAULT_DATE_FROM = 1744220500  # Valeur par défaut si aucun checkpoint n'existe

def load_checkpoint():
    """
    Charge le checkpoint depuis le fichier.
    Si le fichier n'existe pas, on le crée avec DEFAULT_DATE_FROM.
    """
    if not os.path.exists(CHECKPOINT_FILE):
        logging.info(f"Fichier de checkpoint introuvable. Création avec la valeur par défaut : {DEFAULT_DATE_FROM}")
        update_checkpoint(DEFAULT_DATE_FROM)
        return DEFAULT_DATE_FROM
    try:
        with open(CHECKPOINT_FILE, "r") as f:
            checkpoint_str = f.read().strip()
            if checkpoint_str:
                return int(checkpoint_str)
    except Exception as e:
        logging.error(f"Erreur lors de la lecture du checkpoint : {e}")
    return DEFAULT_DATE_FROM

def update_checkpoint(timestamp):
    """
    Enregistre le checkpoint (timestamp) dans le fichier.
    """
    try:
        with open(CHECKPOINT_FILE, "w") as f:
            f.write(str(timestamp))
    except Exception as e:
        logging.error(f"Erreur lors de l'écriture du checkpoint : {e}")

def fetch_data_from_api(url):
    """
    Effectue une requête GET et renvoie le JSON si la réponse est correcte.
    En cas d'erreur, lève une exception.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Lève une erreur pour les réponses 400/500
        return response.json()
    except requests.RequestException as e:
        logging.error(f"Erreur lors de la récupération des données de l'API : {e}")
        raise

def adaptive_fetch_data(endpoint, date_from, initial_step, min_step=500):
    """
    Essaie d'appeler l'API pour la plage [date_from, date_from + step].
    Si l'API renvoie une erreur 400, le pas est réduit par incréments de 500 secondes
    jusqu'à atteindre min_step ou obtenir une réponse valide.
    
    Renvoie un tuple : (données, date_to, step_utilisé)
    """
    step = initial_step
    while True:
        date_to = date_from + step
        url = f"{endpoint}?dateFrom={date_from}&dateTo={date_to}&sportIds=85&lng=en&ref=55&gr=654"
        try:
            logging.info(f"Tentative de requête : dateFrom={date_from} dateTo={date_to} (step={step} sec)")
            data = fetch_data_from_api(url)
            return data, date_to, step
        except requests.RequestException:
            if step > min_step:
                step -= 500
                logging.warning(f"Réduction de la plage à {step} secondes pour dateFrom {date_from}")
            else:
                logging.error(f"Step minimal atteint pour dateFrom {date_from}. Erreur persistante.")
                return None, date_to, step

def process_and_insert_data(cursor, data):
    """
    Traitement des données reçues de l'API et insertion dans la base.
    Cette fonction doit être adaptée aux besoins réels.
    Ici, nous faisons un traitement de base pour compter les enregistrements.
    """
    transfer_count = 0
    if 'items' in data:
        for item in data['items']:
            # Extraction sécurisée des valeurs
            data_id = item.get('id')
            score = item.get('score', '0:0')
            videos = json.dumps(item.get('videos', []))
            equipe1 = item.get('opp1', 'Inconnu')
            equipe2 = item.get('opp2', 'Inconnu')
            
            try:
                primary_score = score.split(' ')[0]
                score_parts = primary_score.split(':')
                score_equipe1 = int(score_parts[0])
                score_equipe2 = int(score_parts[1])
            except (ValueError, IndexError) as e:
                logging.error(f"Erreur dans le format du score : {e}")
                score_equipe1, score_equipe2 = 0, 0

            if '(' in score and ')' in score:
                half_scores = score.split('(')[1].split(')')[0]
                if ',' in half_scores:
                    half_scores = half_scores.split(',')
                    total_1Mt = sum(map(int, half_scores[0].split(':')))
                    total_2Mt = sum(map(int, half_scores[1].split(':')))
                else:
                    total_1Mt, total_2Mt = 0, 0
            else:
                total_1Mt = score_equipe1
                total_2Mt = score_equipe2

            total_score = score_equipe1 + score_equipe2

            # Insertion ou mise à jour selon la logique de votre application
            # Ici, nous supprimons d'abord l'ancien enregistrement puis insérons le nouveau
            cursor.execute("DELETE FROM scores WHERE dataId = %s", (data_id,))
            cursor.execute("""
                INSERT INTO scores (dataId, videos, score, total_score, score_equipe1, score_equipe2, 
                                    total_1Mt, total_2Mt, equipe1, equipe2)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (data_id, videos, score, total_score, score_equipe1, score_equipe2, total_1Mt, total_2Mt, equipe1, equipe2))
            transfer_count += 1
        logging.info(f"Nombre de transferts effectués dans ce segment : {transfer_count}")
    else:
        logging.warning("La clé 'items' n'existe pas dans la réponse de l'API.")
    return transfer_count

def main():
    try:
        db = mysql.connector.connect(**db_config)
        cursor = db.cursor()
        logging.info("Connexion obtenue depuis le pool.")
    except mysql.connector.Error as err:
        logging.error(f"Erreur de connexion : {err}")
        return

    # Suppression des anciennes données de la table "scores"
    cursor.execute("DELETE FROM scores")
    logging.info("Suppression des anciennes données dans la table 'scores'.")

    # Chargement du checkpoint pour déterminer la date de départ
    date_from = load_checkpoint()
    initial_step = 2 * 86400  # Plage initiale de 2 jours en secondes
    now_timestamp = datetime.now().timestamp()
    endpoint = "https://1xbet.cm/service-api/result/web/api/v1/champs"

    total_transfers = 0
    while date_from < now_timestamp:
        data, date_to, used_step = adaptive_fetch_data(endpoint, date_from, initial_step)
        if data is not None:
            transfers = process_and_insert_data(cursor, data)
            total_transfers += transfers
            logging.info(f"Transfer count mis à jour : {total_transfers}")
        else:
            logging.warning(f"Aucune donnée obtenue pour dateFrom {date_from} malgré la réduction du step.")

        # Mise à jour du checkpoint pour persister la dernière date traitée
        update_checkpoint(date_to)
        date_from = date_to

    db.commit()
    cursor.close()
    db.close()
    logging.info("Traitement terminé et les données ont été enregistrées.")

if __name__ == "__main__":
    main()
