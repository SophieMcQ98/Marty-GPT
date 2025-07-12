import sqlite3

conn = sqlite3.connect('/Users/sophie/Library/Messages/chat.db')
cursor = conn.cursor()

cursor.execute("SELECT DISTINCT id FROM handle")
handles = cursor.fetchall()

for h in handles:
    print(h[0])

conn.close()