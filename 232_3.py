import os
from dotenv import load_dotenv
import mysql.connector
import logging

# Charger les variables d'environnement pour sécuriser les identifiants
load_dotenv()

# Configuration du journal
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paramètres de connexion pour les bases de données via le fichier .env
db_config_source = {
    'host': os.getenv('DB_HOST_SOURCE', 'localhost'),
    'user': os.getenv('DB_USER_SOURCE', 'root'),
    'password': os.getenv('DB_PASSWORD'),
    'database': os.getenv('DB_DATABASE_SOURCE', 'sport_scores')
}

db_config_dest = {
    'host': os.getenv('DB_HOST_DEST', 'localhost'),
    'user': os.getenv('DB_USER_DEST', 'root'),
    'password': os.getenv('DB_PASSWORD'),
    'database': os.getenv('DB_DATABASE_DEST', 'mlflow_db')
}

def insert_new_data():
    """Insère dans la base mlflow_db les enregistrements de scores qui ne sont pas présents dans matchs."""
    try:
        # Connexion à la base de destination mlflow_db
        db_dest = mysql.connector.connect(**db_config_dest)
        cursor_dest = db_dest.cursor()
        logging.info("Connexion à la base de données mlflow_db effectuée.")

        # Requête pour insérer les nouvelles données (si dataId n'existe pas dans matchs)
        insert_query = """
            INSERT INTO matchs (dataId, score, total_score, score_equipe1, score_equipe2, total_1Mt, total_2Mt, equipe1, equipe2)
            SELECT s.dataId, s.score, s.total_score, s.score_equipe1, s.score_equipe2, s.total_1Mt, s.total_2Mt, s.equipe1, s.equipe2
            FROM sport_scores.scores s
            LEFT JOIN mlflow_db.matchs m ON s.dataId = m.dataId
            WHERE m.dataId IS NULL
        """
        cursor_dest.execute(insert_query)
        db_dest.commit()
        logging.info(f"Nombre d'enregistrements insérés : {cursor_dest.rowcount}")
        
    except mysql.connector.Error as err:
        logging.error(f"Erreur lors de l'insertion des nouvelles données : {err}")
    
    finally:
        try:
            if cursor_dest:
                cursor_dest.close()
            if db_dest:
                db_dest.close()
        except Exception as e:
            logging.error(f"Erreur lors de la fermeture des connexions : {e}")
        
        logging.info("Insertion des nouvelles données terminée dans mlflow_db.")

if __name__ == "__main__":
    insert_new_data()
