from pydantic import BaseModel

# dummy data
data = {
    "name": "ichigo",
    "age": "19",
    "course": "sword art",
    "ratings": [4, 4.6, "5", "4.5", 4]
}

class Instructor(BaseModel):
    name: str
    age: int
    course: str
    ratings: list[float] = []


user = Instructor(**data)

print(f"Found a Instructor: {user}")
