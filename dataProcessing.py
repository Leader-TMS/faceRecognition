import sqlite3
from datetime import datetime
import requests

def getEmployeesByCode(employeeCode):
    conn = sqlite3.connect('employee.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute('''
        SELECT employee_code, full_name FROM employees 
        WHERE employee_code = ? AND deleted_at IS NULL
        LIMIT 1
    ''', (employeeCode,))

    employee = cursor.fetchone()

    conn.close()
    return employee

def getEmployeesByRFID(rfid):
    conn = sqlite3.connect('employee.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute('''
        SELECT employee_code, full_name FROM employees 
        WHERE rf_id = ? AND deleted_at IS NULL
        LIMIT 1
    ''', (rfid,))

    employee = cursor.fetchone()

    conn.close()

    return employee

def saveAttendance(checkBy, employeeCode, uniqueId):
    try:
        conn = sqlite3.connect('employee.db')
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute('''
            SELECT id FROM employees 
            WHERE employee_code = ? AND deleted_at IS NULL
            LIMIT 1
        ''', (employeeCode,))

        employee = cursor.fetchone()
        if employee:
            employeeId = employee['id']
            createdAt = updatedAt = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            cursor.execute('''
                INSERT INTO attendance (employee_id, created_at, updated_at, deleted_at)
                VALUES (?, ?, ?, ?)
            ''', (employeeId, createdAt, updatedAt, None)) 
            conn.commit()
            conn.close()

            url = 'http://api.tanthanhdat.local/api/post/run/job/save-attendance-job'
            dataSave = {
                "employee_id": employeeId,
                "face_check_in": 1 if checkBy == "face" else 0,
                "rfid_check_in": 0 if checkBy == "face" else 1,
                "creator_id": 1,
                "unique_id": uniqueId
            }

            requests.post(url, data = dataSave)
    except Exception as e:
        return False
    return True

if __name__ == "__main__":
    conn = sqlite3.connect('employee.db')

    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS employees (
            id INTEGER PRIMARY KEY,
            employee_code TEXT NOT NULL UNIQUE,
            rf_id TEXT,
            full_name TEXT NOT NULL,
            birthday DATE NOT NULL,
            sex TEXT CHECK(sex IN ('M', 'F', 'Other')),
            email TEXT,
            phone TEXT,
            created_at DATE NOT NULL,
            updated_at DATE NOT NULL,
            deleted_at DATE
        )
    ''')
    employees = [
        ('001', '0496191378', 'Phạm Ngọc Minh', '2000-02-13', 'M', 'pnminh1302@gmail.com', '0123456789', datetime.now().strftime("%Y-%m-%dT%H:%M:%S"), datetime.now().strftime("%Y-%m-%dT%H:%M:%S"), None),
        ('002', '2188600829', 'Nguyễn Quốc Anh', '2001-01-18', 'M', 'nqanh040101@gmail.com', '0987654321', datetime.now().strftime("%Y-%m-%dT%H:%M:%S"), datetime.now().strftime("%Y-%m-%dT%H:%M:%S"), None),
        ('003', 'RF003', 'Nguyễn Hoàng Linh Dương', '1995-09-12', 'F', 'luatduong95@gmail.com', '0112233445', datetime.now().strftime("%Y-%m-%dT%H:%M:%S"), datetime.now().strftime("%Y-%m-%dT%H:%M:%S"), None),
        ('004', 'RF004', 'Nguyễn Chí Tài', '1990-02-15', 'M', 'info@tms-s.vn', '0123344556', datetime.now().strftime("%Y-%m-%dT%H:%M:%S"), datetime.now().strftime("%Y-%m-%dT%H:%M:%S"), None),
        ('005', 'RF004', 'Trần Trọng Khiêm', '1994-11-21', 'M', 'khiem.trantrong94@gmail.com', '0123344556', datetime.now().strftime("%Y-%m-%dT%H:%M:%S"), datetime.now().strftime("%Y-%m-%dT%H:%M:%S"), None),
        ('006', 'RF004', 'Nguyễn Quốc Hưng', '1997-12-12', 'M', 'nguyenquochung0797@gmail.com', '0123344556', datetime.now().strftime("%Y-%m-%dT%H:%M:%S"), datetime.now().strftime("%Y-%m-%dT%H:%M:%S"), None),
        ('007', 'RF004', 'Võ Lư Thanh Ngân', '1996-10-15', 'F', 'ngannganthanh.1510@gmail.com', '0123344556', datetime.now().strftime("%Y-%m-%dT%H:%M:%S"), datetime.now().strftime("%Y-%m-%dT%H:%M:%S"), None),
        ('008', 'RF004', 'Võ Minh Thông', '2002-12-26', 'M', 'thong89x@gmail.com', '0123344556', datetime.now().strftime("%Y-%m-%dT%H:%M:%S"), datetime.now().strftime("%Y-%m-%dT%H:%M:%S"), None),
        ('009', 'RF004', 'Nguyễn Quang Huy', '2002-05-17', 'M', 'nqhuy1705@gmail.com', '0123344556', datetime.now().strftime("%Y-%m-%dT%H:%M:%S"), datetime.now().strftime("%Y-%m-%dT%H:%M:%S"), None),
        ('010', 'RF004', 'Phạm Trần Anh Tuấn', '2003-04-04', 'M', 'tuananh4403@gmail.com', '0123344556', datetime.now().strftime("%Y-%m-%dT%H:%M:%S"), datetime.now().strftime("%Y-%m-%dT%H:%M:%S"), None),
        ('011', 'RF004', 'Nguyễn Lê Chi', '2003-01-08', 'F', 'nguyenlchi220602@gmail.com', '0123344556', datetime.now().strftime("%Y-%m-%dT%H:%M:%S"), datetime.now().strftime("%Y-%m-%dT%H:%M:%S"), None),
        ('012', 'RF004', 'Nguyễn Vũ Luân', '2001-03-04', 'M', 'ngtranvuluan@gmail.com', '0123344556', datetime.now().strftime("%Y-%m-%dT%H:%M:%S"), datetime.now().strftime("%Y-%m-%dT%H:%M:%S"), None),
        ('013', 'RF004', 'Huỳnh Khánh Tuyên', '2004-05-25', 'F', 'daolam1568@gmail.com', '0123344556', datetime.now().strftime("%Y-%m-%dT%H:%M:%S"), datetime.now().strftime("%Y-%m-%dT%H:%M:%S"), None),
        ('014', 'RF004', 'Nguyễn Văn Hiếu', '2004-05-25', 'M', 'hieunguyen130701iuh@gmail.com', '0123344556', datetime.now().strftime("%Y-%m-%dT%H:%M:%S"), datetime.now().strftime("%Y-%m-%dT%H:%M:%S"), None),
        ('015', 'RF004', 'Trần Mạc Anh Tuyên', '2004-05-25', 'M', 'acetuhoang@gmail.com', '0123344556', datetime.now().strftime("%Y-%m-%dT%H:%M:%S"), datetime.now().strftime("%Y-%m-%dT%H:%M:%S"), None),
    ]

    cursor.executemany('''
        INSERT OR IGNORE INTO employees (employee_code, rf_id, full_name, birthday, sex, email, phone, created_at, updated_at, deleted_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', employees)

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id INTEGER,
            created_at DATE NOT NULL,
            updated_at DATE NOT NULL,
            deleted_at DATE,
            FOREIGN KEY (employee_id) REFERENCES employees(id)
        )
    ''')

    sql = "SELECT * FROM employees"
    recs = cursor.execute(sql)
    for row in recs:
        print(row)

    conn.commit()
    conn.close()

    print("Table 'orders' created successfully.")