from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.prediction import router as prediction_router


app = FastAPI(
    title="Gayathri's Loan Approval ML Application",
    description="Loan approval prediction using Logistic Regression",
    version="1.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


app.mount(
    "/static",
    StaticFiles(directory="app/static"),
    name="static"
)


templates = Jinja2Templates(
    directory="app/templates"
)


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )


@app.get("/prediction", response_class=HTMLResponse)
def prediction_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="prediction.html"
    )


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="dashboard.html"
    )


@app.get("/health")
def health_check():
    return {
        "status": "Application is running"
    }


app.include_router(prediction_router)