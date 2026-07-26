import sqlite3

DB_NAME = "medical_history.db"


# -------------------------------
# Initialize Database
# -------------------------------
def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS prediction_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT,
        disease TEXT,
        confidence REAL,
        prediction_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()


# -------------------------------
# Save Prediction
# -------------------------------
def save_prediction(filename, disease, confidence):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO prediction_history
        (filename, disease, confidence)
        VALUES (?, ?, ?)
    """, (filename, disease, confidence))

    conn.commit()
    conn.close()


# -------------------------------
# Get Prediction History
# -------------------------------
def get_history():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            id,
            filename,
            disease,
            confidence,
            prediction_time
        FROM prediction_history
        ORDER BY id DESC
    """)

    rows = cursor.fetchall()

    conn.close()

    return rows


# -------------------------------
# Dashboard Analytics
# -------------------------------

def get_total_predictions():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT COUNT(*)
        FROM prediction_history
    """)

    total = cursor.fetchone()[0]

    conn.close()

    return total


def get_normal_cases():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT COUNT(*)
        FROM prediction_history
        WHERE disease='NORMAL'
    """)

    count = cursor.fetchone()[0]

    conn.close()

    return count


def get_pneumonia_cases():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT COUNT(*)
        FROM prediction_history
        WHERE disease='PNEUMONIA'
    """)

    count = cursor.fetchone()[0]

    conn.close()

    return count


def get_latest_prediction():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT prediction_time
        FROM prediction_history
        ORDER BY id DESC
        LIMIT 1
    """)

    row = cursor.fetchone()

    conn.close()

    if row:
        return row[0]

    return "No predictions yet"