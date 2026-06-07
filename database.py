import sqlite3

conn = sqlite3.connect("plant_history.db")

cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plant TEXT,
    disease TEXT,
    confidence REAL,
    status TEXT,
    severity TEXT
)
""")

conn.commit()
conn.close()

print("Database c reated successfully!")