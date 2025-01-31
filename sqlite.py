import sqlite3

conn = sqlite3.connect('example.db')
cursor = conn.cursor()

cursor.execute('''CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)''')

cursor.execute('''INSERT INTO users (name, age) VALUES (?, ?)''', ("Alice", 30))

cursor.execute('''SELECT * FROM users''')
print(cursor.fetchall())

conn.commit()
conn.close()
