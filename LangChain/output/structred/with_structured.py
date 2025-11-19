from langchain_google_genai import ChatGoogleGenerativeAI
import dotenv 
from pydantic import BaseModel, Field,EmailStr
from typing import Optional,Literal

dotenv.load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)


class Review(BaseModel):
    key_themes: list[str] = Field(description="Main topics discussed in the review")
    summary: str = Field(description="Short recap of the overall dining experience")
    sentiment: Literal["pos", "neut", "negs"] = Field(description="Overall tone such as positive, neutral, or negative")
    pros :str = Field(description="Positive aspects mentioned in  the review ")
    cons : Optional[str] = Field(default= None,description="Negatives aspects mentioned in the review")
    name : Optional[str] = Field(default= None,description="Name of the reviewer")
    
    email : Optional[EmailStr] = Field(default= None,description="Email of the reviewer")
    cgpa : int =  Field(gt=0 ,lt=10)
    
    
structure = model.with_structured_output(Review)
result =  structure.invoke(""" I had the most incredible dining experience at this restaurant! The food was absolutely delicious - every dish was perfectly prepared and beautifully presented. Our server was attentive, knowledgeable, and made excellent recommendations. The atmosphere was elegant yet comfortable. I cant wait to come back and try more items from their menu. Definitely a new favorite spot!""") 

print(result)
