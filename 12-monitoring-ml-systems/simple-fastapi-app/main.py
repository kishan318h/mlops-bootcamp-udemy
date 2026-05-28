from fastapi import FastAPI
import os
import uvicorn
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI()
port = int(os.environ.get("PORT", 8005))
# for, my_first_api = FastAPI()
# terminal command: uvicorn main:my_first_api --reload

@app.get('/') # path operation decorator
async def root():
    return {"message": "Hello World from FastAPI!"}


@app.get('/demo') # path operation decorator
def demo_function():
    return {"message": "Output from demo function"}


@app.post('/post_demo') # path operation decorator
async def demo_post():
    return {"message": "Output from post demo function"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)

Instrumentator().instrument(app).expose(app)

# get: to read data
# put: to update data
# post: to create data
# delete: to delete data