import sqlite3

def create_connection():
    conn = sqlite3.connect("mood_data.db", check_same_thread=False)
    return conn

def create_table(conn):
    conn.execute('''
        CREATE TABLE IF NOT EXISTS mood_tracker (
            employee_id TEXT,
            text TEXT,
            emotion TEXT,
            confidence REAL,
            stress_score INTEGER,
            date TEXT
        )
    ''')
    conn.commit()
