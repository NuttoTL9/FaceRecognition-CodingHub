#!/usr/bin/env python3
"""
Debug script สำหรับตรวจสอบการเชื่อมต่อ ERPNext
"""
import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_env_variables():
    """ตรวจสอบ environment variables ที่จำเป็น"""
    print("=== ตรวจสอบ Environment Variables ===")
    
    required_vars = [
        'FRAPPE_URL',
        'FRAPPE_API_KEY', 
        'FRAPPE_API_SECRET',
        'POSTGRES_USER',
        'POSTGRES_PASSWORD',
        'POSTGRES_HOST',
        'POSTGRES_DB'
    ]
    
    missing_vars = []
    for var in required_vars:
        value = os.getenv(var)
        if not value or value.startswith('your_'):
            missing_vars.append(var)
            print(f"❌ {var}: ไม่ได้กำหนดค่าหรือใช้ค่า default")
        else:
            print(f"✅ {var}: {'*' * len(value)}")  # Hide sensitive values
    
    return len(missing_vars) == 0, missing_vars

def test_erpnext_connection():
    """ทดสอบการเชื่อมต่อ ERPNext"""
    print("\n=== ทดสอบการเชื่อมต่อ ERPNext ===")
    
    frappe_url = os.getenv('FRAPPE_URL')
    frappe_api_key = os.getenv('FRAPPE_API_KEY')
    frappe_api_secret = os.getenv('FRAPPE_API_SECRET')
    
    if not all([frappe_url, frappe_api_key, frappe_api_secret]):
        print("❌ ERPNext configuration ไม่ครบถ้วน")
        return False
    
    try:
        # Test basic connection
        url = f"{frappe_url}/api/resource/Employee"
        params = {
            "fields": '["name"]',
            "limit": 1
        }
        auth = (frappe_api_key, frappe_api_secret)
        
        print(f"🔗 กำลังเชื่อมต่อ: {frappe_url}")
        response = requests.get(url, params=params, auth=auth, timeout=10)
        
        if response.ok:
            print("✅ เชื่อมต่อ ERPNext สำเร็จ")
            data = response.json()
            print(f"📊 พบ Employee จำนวน: {len(data.get('data', []))}")
            return True
        else:
            print(f"❌ เชื่อมต่อ ERPNext ล้มเหลว: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ ข้อผิดพลาดในการเชื่อมต่อ ERPNext: {e}")
        return False

def test_postgres_connection():
    """ทดสอบการเชื่อมต่อ PostgreSQL"""
    print("\n=== ทดสอบการเชื่อมต่อ PostgreSQL ===")
    
    try:
        import asyncpg
        import asyncio
        
        async def connect_db():
            user = os.getenv('POSTGRES_USER')
            password = os.getenv('POSTGRES_PASSWORD')
            host = os.getenv('POSTGRES_HOST', 'localhost')
            port = os.getenv('POSTGRES_PORT', '5432')
            database = os.getenv('POSTGRES_DB')
            
            if not all([user, password, database]):
                print("❌ PostgreSQL configuration ไม่ครบถ้วน")
                return False
            
            try:
                conn = await asyncpg.connect(
                    user=user,
                    password=password,
                    host=host,
                    port=port,
                    database=database
                )
                await conn.close()
                print("✅ เชื่อมต่อ PostgreSQL สำเร็จ")
                return True
            except Exception as e:
                print(f"❌ เชื่อมต่อ PostgreSQL ล้มเหลว: {e}")
                return False
        
        return asyncio.run(connect_db())
        
    except ImportError:
        print("❌ asyncpg ไม่ได้ติดตั้ง")
        return False

def test_log_event_endpoint():
    """ทดสอบ log_event endpoint"""
    print("\n=== ทดสอบ Log Event Endpoint ===")
    
    try:
        url = "http://localhost:8000/log_event/"
        payload = {
            "name": "TEST001",
            "event": "in"
        }
        
        response = requests.post(url, json=payload, timeout=5)
        
        if response.ok:
            print("✅ Log Event endpoint ทำงานปกติ")
            print(f"Response: {response.json()}")
            return True
        else:
            print(f"❌ Log Event endpoint ล้มเหลว: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ FastAPI server ไม่ได้ทำงาน (Connection refused)")
        return False
    except Exception as e:
        print(f"❌ ข้อผิดพลาดในการทดสอบ endpoint: {e}")
        return False

def main():
    print("🔍 Face Recognition ERPNext Debug Tool")
    print("=" * 50)
    
    # Check environment variables
    env_ok, missing_vars = check_env_variables()
    
    # Test connections
    erpnext_ok = test_erpnext_connection() if env_ok else False
    postgres_ok = test_postgres_connection() if env_ok else False
    endpoint_ok = test_log_event_endpoint()
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 สรุปผลการตรวจสอบ:")
    print(f"Environment Variables: {'✅' if env_ok else '❌'}")
    print(f"ERPNext Connection: {'✅' if erpnext_ok else '❌'}")
    print(f"PostgreSQL Connection: {'✅' if postgres_ok else '❌'}")
    print(f"FastAPI Endpoint: {'✅' if endpoint_ok else '❌'}")
    
    if not env_ok:
        print(f"\n❗ กรุณากำหนดค่าตัวแปรเหล่านี้ในไฟล์ .env:")
        for var in missing_vars:
            print(f"   - {var}")
    
    if not endpoint_ok:
        print(f"\n❗ กรุณาเริ่ม FastAPI server:")
        print(f"   cd FastAPI && python3 -m uvicorn main_fastapi:app --host 0.0.0.0 --port 8000")
    
    all_ok = env_ok and erpnext_ok and postgres_ok and endpoint_ok
    print(f"\n🎯 สถานะรวม: {'✅ ทุกอย่างพร้อมใช้งาน' if all_ok else '❌ มีปัญหาที่ต้องแก้ไข'}")

if __name__ == "__main__":
    main()