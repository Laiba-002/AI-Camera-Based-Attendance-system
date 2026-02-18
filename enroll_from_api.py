"""
API-based Employee Enrollment for FastAPI Standalone
Fetches employee data from API and generates face embeddings using InsightFace ONNX
"""
import os
import cv2
import numpy as np
import pickle
import requests
import base64
from io import BytesIO
from PIL import Image
from typing import List, Dict, Optional, Tuple
import config
from database import DatabaseManager
from vector_db import VectorDBManager
from face_recognition_engine_onnx import FaceRecognitionEngineONNX


class APIEnrollment:
    """Handles employee enrollment from API using InsightFace ONNX"""

    def __init__(self, api_url: str = None):
        self.db = DatabaseManager()
        self.vector_db = VectorDBManager()
        self.api_url = api_url or config.API_ENDPOINT

        # Initialize InsightFace ONNX engine (same as production system)
        print("Initializing InsightFace ONNX engine...")
        self.face_engine = FaceRecognitionEngineONNX()
        print(f"✓ API Enrollment initialized (InsightFace ONNX - {config.ONNX_MODEL_NAME})")

    def fetch_employees_from_api(self) -> Optional[List[Dict]]:
        """Fetch employee data from API"""
        try:
            print(f"\nFetching employee data from API...")
            print(f"API URL: {self.api_url}")

            response = requests.get(self.api_url, timeout=30)
            response.raise_for_status()

            data = response.json()

            # Handle API response structure: {"status": true, "message": "...", "data": [...]}
            if not data.get('status'):
                print(
                    f"✗ API returned status false: {data.get('message', 'Unknown error')}")
                return None

            employees = data.get('data', [])
            message = data.get('message', '')

            print(f"✓ {message}")
            print(
                f"✓ Successfully fetched {len(employees)} employee(s) from API")
            return employees

        except requests.exceptions.RequestException as e:
            print(f"✗ Error fetching data from API: {str(e)}")
            return None
        except Exception as e:
            print(f"✗ Error processing API response: {str(e)}")
            return None

    def decode_image_from_base64(self, base64_string: str) -> Optional[np.ndarray]:
        """Decode base64 image string to OpenCV image"""
        try:
            # Remove data URL prefix if present
            if ',' in base64_string:
                base64_string = base64_string.split(',', 1)[1]

            # Decode base64 to bytes
            image_bytes = base64.b64decode(base64_string)

            # Convert bytes to PIL Image
            pil_image = Image.open(BytesIO(image_bytes))

            # Convert PIL to OpenCV format (BGR)
            opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

            return opencv_image

        except Exception as e:
            print(f"  ✗ Error decoding base64 image: {str(e)}")
            return None

    def download_image_from_url(self, image_url: str) -> Optional[np.ndarray]:
        """Download image from URL and convert to OpenCV format"""
        try:
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()

            # Convert bytes to PIL Image
            pil_image = Image.open(BytesIO(response.content))

            # Convert PIL to OpenCV format (BGR)
            opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

            return opencv_image

        except Exception as e:
            print(f"  ✗ Error downloading image from URL: {str(e)}")
            return None

    def generate_face_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Generate face embedding using InsightFace ONNX (same as production)"""
        try:
            if image is None or image.size == 0:
                return None

            # Use InsightFace ONNX to detect and generate embedding
            # enrollment_mode=True to accept high-quality close-up photos
            faces = self.face_engine.detect_faces(image, enrollment_mode=True)

            if faces and len(faces) > 0:
                # Use the first detected face
                embedding = faces[0]['embedding']
                return embedding

            return None

        except Exception as e:
            # Don't print error for every image, just return None
            return None

    def enroll_employee(self, employee_data: Dict, force_reenroll: bool = False) -> bool:
        """
        Enroll a single employee from API data

        Expected API format: {
            "employeeId": int,
            "employeeCode": str,
            "employeeName": str,
            "attendancePics": [{"imageUrl": str, "createdOn": str}, ...]
        }
        """
        try:
            # Extract employee information
            employee_id = str(employee_data.get('employeeId', ''))
            employee_code = employee_data.get('employeeCode', '')
            name = employee_data.get('employeeName', 'Unknown')
            attendance_pics = employee_data.get('attendancePics', [])

            if not employee_id:
                print(f"  ✗ {name}: Missing employee ID")
                return False

            if not attendance_pics:
                print(f"  ✗ {name}: No attendance pictures found")
                return False

            # Check if already enrolled (use employee_code as unique identifier)
            if self.vector_db.employee_exists(employee_code) and not force_reenroll:
                print(
                    f"  ⊙ {name} (Code: {employee_code}): Already enrolled, skipping")
                return False

            # If force re-enroll, delete old embeddings first
            if force_reenroll and self.vector_db.employee_exists(employee_code):
                print(
                    f"  ⟳ {name} (Code: {employee_code}): Deleting old embeddings...")
                self.vector_db.delete_face_encoding(
                    employee_code)  # Uses employee_code as ID

            print(
                f"\n  → Processing: {name} (Code: {employee_code}, ID: {employee_id})")
            print(f"    Found {len(attendance_pics)} image(s)")

            # Process all images and collect embeddings
            all_embeddings = []
            first_valid_image = None

            for idx, pic_data in enumerate(attendance_pics, 1):
                image_url = pic_data.get('imageUrl')
                if not image_url:
                    continue

                print(
                    f"    → Processing image {idx}/{len(attendance_pics)}...")
                opencv_image = self.download_image_from_url(image_url)

                if opencv_image is None:
                    print(f"      ✗ Failed to download image {idx}")
                    continue

                # Generate face embedding
                face_embedding = self.generate_face_embedding(opencv_image)

                if face_embedding is None:
                    print(f"      ✗ No face detected in image {idx}")
                    continue

                all_embeddings.append(face_embedding)
                if first_valid_image is None:
                    first_valid_image = opencv_image
                print(f"      ✓ Embedding generated for image {idx}")

            if not all_embeddings:
                print(f"    ✗ No valid embeddings generated for {name}")
                return False

            # Average all embeddings for robust recognition
            if len(all_embeddings) > 1:
                print(
                    f"    → Averaging {len(all_embeddings)} embeddings for robust recognition")
                averaged_embedding = np.mean(all_embeddings, axis=0)
            else:
                averaged_embedding = all_embeddings[0]
            
            # CRITICAL: Normalize the embedding after averaging
            # InsightFace provides normalized embeddings, but averaging changes the norm
            embedding_norm = np.linalg.norm(averaged_embedding)
            if embedding_norm > 0:
                averaged_embedding = averaged_embedding / embedding_norm
            
            print(
                f"    ✓ Final embedding created from {len(all_embeddings)} image(s) (normalized)")

            # Serialize embedding for SQLite storage
            encoding_bytes = pickle.dumps(averaged_embedding)

            # Convert first valid image to base64 for storage
            _, buffer = cv2.imencode('.jpg', first_valid_image)
            image_base64 = base64.b64encode(buffer).decode('utf-8')

            # Add to SQLite database (using employee_code as ID for uniqueness)
            conn = self.db.get_connection()
            cursor = conn.cursor()

            cursor.execute('''
                INSERT OR REPLACE INTO employees 
                (employee_id, name, email, department, position, face_encoding, is_active)
                VALUES (?, ?, ?, ?, ?, ?, 1)
            ''', (employee_code, name, None, None, None, encoding_bytes))

            conn.commit()
            conn.close()

            print(f"    ✓ Added to main SQLite database")

            # Add to Vector Database with metadata
            metadata = {
                'employee_id': employee_id,
                'employee_code': employee_code,
                'image': image_base64,
                'image_count': len(all_embeddings)
            }

            self.vector_db.add_employee_embedding(
                employee_id=employee_code,
                name=name,
                face_embedding=averaged_embedding,
                metadata=metadata
            )

            print(f"    ✓ Added to Vector Database")
            print(f"  ✓ Successfully enrolled {name} (Code: {employee_code})")

            return True

        except Exception as e:
            print(f"  ✗ Error enrolling employee: {str(e)}")
            return False

    def enroll_all_from_api(self, skip_existing: bool = True) -> Dict:
        """Enroll all employees from API"""
        print("\n" + "="*70)
        print("Starting API-based Employee Enrollment")
        if skip_existing:
            print("Mode: INCREMENTAL (Skip existing employees)")
        else:
            print("Mode: FULL RE-ENROLLMENT")
        print("="*70)

        # Fetch employees
        employees = self.fetch_employees_from_api()
        if not employees:
            return {
                'success': False,
                'message': 'Failed to fetch employees from API'
            }

        # Check existing employees
        existing_employee_ids = set()
        if skip_existing:
            existing_employee_ids = set(
                self.vector_db.get_existing_employee_ids())
            print(
                f"\n✓ Found {len(existing_employee_ids)} existing employee(s) in database")
            print(f"  Will only process new employees...")

        # Enroll each employee
        total = len(employees)
        enrolled = 0
        skipped = 0
        failed = 0

        print(f"\nProcessing {total} employee(s) from API...\n")

        for emp in employees:
            employee_code = emp.get('employeeCode', '')

            # Check if already exists
            if skip_existing and employee_code in existing_employee_ids:
                employee_name = emp.get('employeeName', 'Unknown')
                print(f"\n➤ Skipping: {employee_name} (Code: {employee_code})")
                print(f"  Reason: Already enrolled in database")
                skipped += 1
                continue

            result = self.enroll_employee(
                emp, force_reenroll=not skip_existing)
            if result:
                enrolled += 1
            else:
                failed += 1

        # Summary
        total_processed = enrolled + failed
        print("\n" + "="*70)
        print("Enrollment Summary:")
        print("="*70)
        print(f"  Total from API: {total}")
        if skip_existing:
            print(f"  Already Enrolled (Skipped): {skipped}")
            print(f"  New Employees Processed: {total_processed}")
        else:
            print(f"  Employees Processed: {total_processed}")
        print(f"  ✓ Successfully Enrolled: {enrolled}")
        print(f"  ✗ Failed: {failed}")
        if total_processed > 0:
            print(f"  Success Rate: {(enrolled/total_processed*100):.1f}%")
        print(f"  Total in Database Now: {self.vector_db.count()}")
        print("="*70)

        return {
            'success': True,
            'total': total,
            'enrolled': enrolled,
            'skipped': skipped,
            'failed': failed
        }


def main():
    """Main enrollment function"""
    import sys

    # Check for --full flag to force re-enrollment
    force_full = '--full' in sys.argv or '--re-enroll' in sys.argv

    if force_full:
        print("\n⚠ Full re-enrollment mode enabled")
        print("  All employees will be re-processed\n")

    try:
        enrollor = APIEnrollment()

        # Show current database state
        current_count = enrollor.vector_db.count()
        print(f"\nCurrent database: {current_count} employee(s) enrolled\n")

        # Enroll
        result = enrollor.enroll_all_from_api(skip_existing=not force_full)

        if result['success']:
            print("\n✓ Enrollment completed!")
            print(f"  Database: {config.DATABASE_PATH}")
            print(f"  Vector DB: {os.path.join(config.BASE_DIR, 'vector_db')}")

            if result['skipped'] > 0 and result['enrolled'] == 0:
                print("\n  ℹ No new employees to enroll. All up to date!")
                print("    Use --full flag to force re-enrollment of all employees\n")
        else:
            print(f"\n✗ Enrollment failed: {result.get('message')}")

    except Exception as e:
        print(f"\n✗ Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
