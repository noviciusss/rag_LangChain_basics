from typing import TypedDict

class person(TypedDict):
    name: str
    age: int
    

new_person = person(name="Alice", age=30)
new_person_invalid = person(name="Bob", age="thirty")  
new_person_1 : person = {"name": "Charlie", "age": 25}

print(new_person)
print(new_person_1["name"])