from routers import apis
from fastapi import FastAPI


app = FastAPI()
app.include_router(apis.router)
