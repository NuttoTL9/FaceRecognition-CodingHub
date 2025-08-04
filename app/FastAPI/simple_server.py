from fastapi import FastAPI
from pydantic import BaseModel
import datetime

app = FastAPI()

class LogData(BaseModel):
    name: str
    event: str

@app.get("/")
async def root():
    return {"message": "Face Recognition API Server"}

@app.post("/log_event/")
async def log_event(data: LogData):
    print(f"Received log event: {data.name} - {data.event} at {datetime.datetime.now()}")
    return {
        "status": "success",
        "message": f"Logged {data.name} - {data.event}",
        "timestamp": datetime.datetime.now().isoformat()
    }

@app.post("/log_event_with_snap/")
async def log_event_with_snap(name: str, event: str, employee_id: str):
    print(f"Received log with snap: {name} ({employee_id}) - {event} at {datetime.datetime.now()}")
    return {
        "status": "success", 
        "message": f"Logged {name} - {event} with snapshot",
        "timestamp": datetime.datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)