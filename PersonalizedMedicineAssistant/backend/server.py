from fastapi import FastAPI
from routes.predict import router as predict_router
import uvicorn

app = FastAPI(title="Personalized Medicine Assistant API")

# Include Routes
app.include_router(predict_router)

@app.get("/")
async def root():
    return {"message": "Welcome to Personalized Medicine Assistant API"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
