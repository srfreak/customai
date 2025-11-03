from fastapi import FastAPI, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
from core.config import settings
from core.auth import verify_token, RoleChecker
from core.database import connect_to_mongo, close_mongo_connection

# Import routers
from apps.agents.sales.strategy_ingest import router as sales_strategy_router
from apps.agents.sales.call_handler import router as sales_call_router
from apps.agents.sales.voice_handler import router as sales_voice_router
from apps.agents.sales.agent import router as sales_agent_router
from apps.users.endpoints import router as user_router
from apps.admin_panel.user_management import router as admin_user_router
from apps.memory.memory_manager import router as memory_router
from apps.integrations.telephony.twilio import router as twilio_router
from apps.ops import router as ops_router
from apps.templates.presets import router as presets_router

app = FastAPI(
    title="Scriza AI Platform",
    description="Next-gen platform for creating, training, and deploying human-like AI agents",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add security
security = HTTPBearer()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Add event handlers for MongoDB
app.add_event_handler("startup", connect_to_mongo)
app.add_event_handler("shutdown", close_mongo_connection)

# Include routers
app.include_router(
    sales_strategy_router,
    prefix="/api/v1/agent/sales",
    tags=["sales-agent"],
    dependencies=[Depends(RoleChecker(["user", "admin"]))]
)

app.include_router(
    sales_call_router,
    prefix="/api/v1/agent/sales",
    tags=["sales-call"],
    dependencies=[Depends(RoleChecker(["user", "admin"]))]
)

app.include_router(
    sales_voice_router,
    prefix="/api/v1/agent/sales",
    tags=["sales-voice"],
    dependencies=[Depends(RoleChecker(["user", "admin"]))]
)

app.include_router(
    sales_agent_router,
    prefix="/api/v1/agent/sales",
    tags=["sales-agent-tools"],
    dependencies=[Depends(RoleChecker(["user", "admin"]))]
)

app.include_router(
    user_router,
    prefix="/api/v1/user",
    tags=["user"]
)

app.include_router(
    admin_user_router,
    prefix="/api/v1/admin",
    tags=["admin"],
    dependencies=[Depends(RoleChecker(["admin"]))]
)

app.include_router(
    memory_router,
    prefix="/api/v1/memory",
    tags=["memory"],
    dependencies=[Depends(RoleChecker(["user", "admin"]))]
)

app.include_router(
    twilio_router,
    prefix="/api/v1/integrations/telephony/twilio",
    tags=["telephony"]
)

app.include_router(
    ops_router,
    tags=["ops"],
)

app.include_router(
    presets_router,
    prefix="/api/v1",
    tags=["templates"],
)

@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Scriza AI Platform is running"}

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Welcome to Scriza AI Platform"}


@app.get("/login", response_class=HTMLResponse)
async def user_login_page(request: Request):
    """Render the end-user login helper template."""
    return templates.TemplateResponse("login.html", {"request": request})


@app.get("/admin/login", response_class=HTMLResponse)
async def admin_login_page(request: Request):
    """Render the admin login helper template."""
    return templates.TemplateResponse("admin_login.html", {"request": request})


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Render the unified dashboard shell (auth handled client-side)."""
    return templates.TemplateResponse("dashboard.html", {"request": request})

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
