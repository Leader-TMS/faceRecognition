# import mysql.connector

# conn = mysql.connector.connect(
#     host="localhost",
#     user="root",
#     password="root",
#     database="ttd_auth_test"
# )

# cursor = conn.cursor()

# # #Tạo bảng mới
# # cursor.execute('''
# # CREATE TABLE IF NOT EXISTS users (
# #     id INT AUTO_INCREMENT PRIMARY KEY,
# #     name VARCHAR(100),
# #     age INT
# # )
# # ''')

# # #Thêm dữ liệu vào bảng
# # cursor.execute('''
# # INSERT INTO users (name, age) VALUES (%s, %s)
# # ''', ("Alice", 30))

# # # Lưu thay đổi vào cơ sở dữ liệu
# # conn.commit()

# cursor.execute('SELECT * FROM users')
# for row in cursor.fetchall():
#     print(row)

# # Đóng kết nối
# cursor.close()
# conn.close()

import mysql.connector
import configparser

config = configparser.ConfigParser()
config.read('config.ini')

host = config['settingMySQL']["HOST"]
port = config['settingMySQL']["PORT"]
user = config['settingMySQL']["USER"]
password = config['settingMySQL']["PASSWORD"]
database = config['settingMySQL']["DATABASE"]

conn = mysql.connector.connect(
    host=host,
    port=port,
    user=user,
    password=password,
    database=database
)

if conn.is_connected():
    print("Kết nối thành công!")

cursor = conn.cursor()
cursor.execute('SELECT * FROM users')
for row in cursor.fetchall():
    print(row)


# Đóng kết nối
cursor.close()
conn.close()