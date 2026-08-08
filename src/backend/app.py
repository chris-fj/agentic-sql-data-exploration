import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from logging_config import setup_logging

from .api.echo import router as echo_router
from .api.llm import router as llm_router
from .api.sql_agent import router as sql_agent_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    yield


logger = logging.getLogger(__name__)

app = FastAPI(title="SQL Agent Backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(echo_router, prefix="/api")
app.include_router(llm_router, prefix="/api")
app.include_router(sql_agent_router, prefix="/api")
