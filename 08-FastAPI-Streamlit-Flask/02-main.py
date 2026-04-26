from fastapi import FastAPI

app = FastAPI()
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


# get: to read data
# put: to update data
# post: to create data
# delete: to delete data