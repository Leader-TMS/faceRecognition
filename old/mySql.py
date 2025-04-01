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

# host = config['settingMySQL']["HOST"]
# port = config['settingMySQL']["PORT"]
# user = config['settingMySQL']["USER"]
# password = config['settingMySQL']["PASSWORD"]
# database = config['settingMySQL']["DATABASE"]

host = config['stagingSettingMySQL']["HOST"]
port = config['stagingSettingMySQL']["PORT"]
user = config['stagingSettingMySQL']["USER"]
password = config['stagingSettingMySQL']["PASSWORD"]
database = config['stagingSettingMySQL']["DATABASE"]

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

# EXPORT PROD
# import mysql.connector
# import csv

# conn = mysql.connector.connect(
#     host="103.92.30.66",
#     user="ttd",
#     port=33064,
#     password="M8C,[KZok9]4",
#     database="ttd_data"
# )
# cursor = conn.cursor()

# query = """
# SELECT name, hat_model, color, logo, quantity, po, started_at, completion_date 
# FROM customer_orders 
# WHERE status LIKE 'approval' 
# AND date_approval >= '2024-12-01' 
# AND deleted_at IS NULL;
# """
# cursor.execute(query)

# columns = [desc[0] for desc in cursor.description]

# with open('customer_orders_data.csv', 'w', newline='', encoding='utf-8') as file:
#     writer = csv.writer(file)
#     writer.writerow(columns)
#     for row in cursor:
#         print(f'row: {row}')
#         writer.writerow(row)

# cursor.close()
# conn.close()

# print("Dữ liệu đã được xuất ra file customer_orders_data.csv")