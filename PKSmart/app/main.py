from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from . import models, database

from .routers import auth, projects, analysis, cohorts

# Initialize Database Tables
models.Base.metadata.create_all(bind=database.engine)

app = FastAPI(title="PKSmart Platform")

# Add Session Middleware for Global Context (PK Params etc)
# Secret key should be in env vars in production
app.add_middleware(SessionMiddleware, secret_key="YOUR_SUPER_SECRET_KEY")

# Mount static files (CSS, JS, Images)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Templates
templates = Jinja2Templates(directory="app/templates")

# Include Routers
app.include_router(auth.router)
app.include_router(projects.router)
app.include_router(analysis.router)
app.include_router(cohorts.router)
from .routers import dashboard
app.include_router(dashboard.router)

@app.get("/")
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
