from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import mpbpk, qsar
import uvicorn

app = FastAPI(
    title="SimuPharma API",
    description="Backend for mPBPK Simulation and QSAR Toxicity Prediction",
    version="1.0.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(mpbpk.router)
app.include_router(qsar.router)

@app.get("/api/health")
async def health_check():
    return {"status": "ok", "service": "SimuPharma API"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
