from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.echo import router as echo_router
from .api.llm import router as llm_router

app = FastAPI(title="SQL Agent Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(echo_router, prefix="/api")
app.include_router(llm_router, prefix="/api")
