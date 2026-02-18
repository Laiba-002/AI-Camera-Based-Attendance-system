"""
Simplified Database Manager for FastAPI
Standalone version with auto-initialization
"""
import sqlite3
import os
from datetime import datetime, date
from typing import List, Dict, Optional
import config


class DatabaseManager:
    """Manages database operations for the attendance system"""

    def __init__(self, db_path: str = config.DATABASE_PATH):
        self.db_path = db_path
        self.initialize_database()

    def initialize_database(self):
        """Initialize database schema with all required tables"""
        conn = self.get_connection()
        cursor = conn.cursor()

        # Employees table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS employees (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                employee_id TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                email TEXT,
                department TEXT,
                position TEXT,
                face_encoding BLOB NOT NULL,
                enrolled_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active INTEGER DEFAULT 1
            )
        ''')

        # Attendance records table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                employee_id TEXT NOT NULL,
                date TEXT NOT NULL,
                punch_type TEXT NOT NULL,
                punch_time TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                notes TEXT,
                FOREIGN KEY (employee_id) REFERENCES employees(employee_id)
            )
        ''')

        # Recognition logs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS recognition_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                employee_id TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                confidence REAL,
                action TEXT,
                success INTEGER,
                notes TEXT
            )
        ''')

        # Cooldown tracking table (persistent across restarts)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS attendance_cooldown (
                employee_id TEXT PRIMARY KEY,
                last_attendance_time TIMESTAMP NOT NULL,
                action_type TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        conn.commit()
        conn.close()
        print(f"✓ Database initialized: {self.db_path}")

    def get_connection(self) -> sqlite3.Connection:
        """Create and return a database connection"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def get_all_employees(self) -> List[Dict]:
        """Get all active employees"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT employee_id, name, email, department, position, 
                   enrolled_date, is_active
            FROM employees 
            WHERE is_active = 1
            ORDER BY name
        ''')
        employees = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return employees

    def get_employee_by_id(self, employee_id: str) -> Optional[Dict]:
        """Get employee details by ID"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT employee_id, name, email, department, position, 
                   enrolled_date, is_active
            FROM employees 
            WHERE employee_id = ?
        ''', (employee_id,))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None

    def get_attendance_records(self, start_date: str = None, end_date: str = None,
                               employee_id: str = None) -> List[Dict]:
        """Get attendance records with filters"""
        conn = self.get_connection()
        cursor = conn.cursor()

        query = '''
            SELECT a.id, a.employee_id, e.name, a.date, a.punch_type, 
                   a.punch_time, a.timestamp, a.notes, e.department, e.position
            FROM attendance a
            JOIN employees e ON a.employee_id = e.employee_id
            WHERE 1=1
        '''
        params = []

        if start_date:
            query += ' AND a.date >= ?'
            params.append(start_date)

        if end_date:
            query += ' AND a.date <= ?'
            params.append(end_date)

        if employee_id:
            query += ' AND a.employee_id = ?'
            params.append(employee_id)

        query += ' ORDER BY a.timestamp DESC'

        cursor.execute(query, params)
        records = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return records

    def get_today_attendance(self) -> List[Dict]:
        """Get today's attendance records"""
        today = date.today().isoformat()
        return self.get_attendance_records(start_date=today, end_date=today)

    def get_employee_today_status(self, employee_id: str) -> Dict:
        """Get employee's attendance status for today"""
        today = date.today().isoformat()
        records = self.get_attendance_records(
            start_date=today, end_date=today, employee_id=employee_id)

        status = {
            'employee_id': employee_id,
            'date': today,
            'checked_in': False,
            'checked_out': False,
            'check_in_time': None,
            'check_out_time': None,
            'records': records
        }

        for record in records:
            if record['punch_type'] == 'check_in':
                status['checked_in'] = True
                if not status['check_in_time'] or record['punch_time'] < status['check_in_time']:
                    status['check_in_time'] = record['punch_time']
            elif record['punch_type'] == 'check_out':
                status['checked_out'] = True
                if not status['check_out_time'] or record['punch_time'] > status['check_out_time']:
                    status['check_out_time'] = record['punch_time']

        return status

    def get_attendance_statistics(self, start_date: str = None, end_date: str = None) -> Dict:
        """Get attendance statistics for a date range"""
        records = self.get_attendance_records(
            start_date=start_date, end_date=end_date)

        stats = {
            'total_records': len(records),
            'unique_employees': len(set(r['employee_id'] for r in records)),
            'check_ins': len([r for r in records if r['punch_type'] == 'check_in']),
            'check_outs': len([r for r in records if r['punch_type'] == 'check_out']),
            'date_range': {
                'start': start_date or 'beginning',
                'end': end_date or 'now'
            }
        }

        return stats

    def check_attendance_cooldown(self, employee_id: str, cooldown_seconds: int = 300) -> bool:
        """
        Check if employee is past cooldown period (persistent across restarts)
        Returns True if can mark attendance, False if in cooldown
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            cursor.execute('''
                SELECT last_attendance_time 
                FROM attendance_cooldown 
                WHERE employee_id = ?
            ''', (employee_id,))

            row = cursor.fetchone()
            conn.close()

            if not row:
                return True  # No previous record, can mark

            last_time = datetime.fromisoformat(row['last_attendance_time'])
            elapsed = (datetime.now() - last_time).total_seconds()

            return elapsed >= cooldown_seconds

        except Exception as e:
            print(f"Cooldown check error: {e}")
            return True  # On error, allow attendance

    def update_attendance_cooldown(self, employee_id: str, action_type: str):
        """Update cooldown timestamp for employee"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            cursor.execute('''
                INSERT OR REPLACE INTO attendance_cooldown 
                (employee_id, last_attendance_time, action_type, updated_at)
                VALUES (?, ?, ?, ?)
            ''', (employee_id, datetime.now().isoformat(), action_type, datetime.now().isoformat()))

            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Cooldown update error: {e}")

    def mark_attendance(self, employee_id: str, punch_type: str,
                        confidence: float = None, liveness_passed: bool = True,
                        notes: str = None) -> tuple:
        """
        Mark attendance for an employee
        Now with confidence and liveness tracking
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            current_date = date.today().isoformat()
            current_time = datetime.now().strftime('%H:%M:%S')

            # Add metadata to notes
            metadata = []
            if confidence is not None:
                metadata.append(f"confidence={confidence:.2%}")
            if not liveness_passed:
                metadata.append("liveness_check_failed")
            if notes:
                metadata.append(notes)

            notes_str = "; ".join(metadata) if metadata else None

            cursor.execute('''
                INSERT INTO attendance (employee_id, date, punch_type, punch_time, notes)
                VALUES (?, ?, ?, ?, ?)
            ''', (employee_id, current_date, punch_type, current_time, notes_str))

            # Update cooldown
            self.update_attendance_cooldown(employee_id, punch_type)

            conn.commit()
            conn.close()

            return True, f"Attendance marked: {punch_type}"
        except Exception as e:
            return False, f"Error marking attendance: {str(e)}"

    def get_recognition_logs(self, limit: int = 100) -> List[Dict]:
        """Get recent recognition logs"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT r.id, r.employee_id, e.name, r.timestamp, r.confidence, 
                   r.action, r.success
            FROM recognition_logs r
            LEFT JOIN employees e ON r.employee_id = e.employee_id
            ORDER BY r.timestamp DESC
            LIMIT ?
        ''', (limit,))
        logs = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return logs
