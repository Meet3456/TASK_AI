from fastapi import FastAPI, HTTPException, Query, Request
from langchain.chat_models.openai import ChatOpenAI
from pydantic import BaseModel
import openai
from langchain.chains import ConversationChain
from fastapi.responses import JSONResponse  # To return JSON responses
from exception import CustomException
import sys

# from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate


from dotenv import load_dotenv
import os

load_dotenv()

class PromptUpdateRequest(BaseModel):
    new_prompt_template: str

# Initialize FastAPI app
app = FastAPI()

# Define a Pydantic model for the request body when using POST
# class PromptUpdateRequest(BaseModel):
#     new_prompt: str


openai.api_key = os.getenv("OPENAI_API_KEY")
# Define the GET endpoint
@app.get("/generate/")
async def generate_response(prompt: str = Query(..., description="The prompt for generating a response")):
    try:
        response = openai.Completion.create(
            model = "gpt-3.5-turbo-instruct",
            prompt=prompt,
            max_tokens=1024,  
            temperature=0.8
        )
        return {"response": response.choices[0].text.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to generate response")

# Using Langchain LLM Chain:
@app.get("/new_generate")
async def generate_response(human_input : str):
    try:
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are a chatbot having a conversation with a human."), # The persistent system prompt
            MessagesPlaceholder(variable_name="chat_history"), # Where the memory will be stored.
            HumanMessagePromptTemplate.from_template("{human_input}"), # Where the human input will injected
        ])
    
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        llm = ChatOpenAI()

        chat_llm_chain = LLMChain(
            llm=llm,
            prompt=prompt,
            verbose=True,
            memory=memory,
        )
        response = chat_llm_chain.predict(human_input=human_input)
        return {"response":response}
    except Exception as e:
        raise CustomException(e,sys)


# @app.post("/update_response")
# async def updated_response(new_prompt:str):

#     try:
#         memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
#         prompt = "you are provided with the previous response generated : {response}, make changes according to the user request.user request = {new_prompt}"
#         prompt = PromptTemplate(
#             template=prompt,
#             input_variables=["response","new_prompt"]
#         )

#         response = generate_response()
#         print(response)
#         # new_prompt += "{chat_history}"
#         # prompt = PromptTemplate(
#         #     template=new_prompt,
#         #     input_variables=["chat_history"]

#         # )
#         llm = ChatOpenAI()
#         chat_llm_chain = LLMChain(
#             llm=llm,
#             prompt=prompt,
#             verbose=True,
#             memory=memory,
#         )
#         response = chat_llm_chain.predict(human_input=new_prompt)
#         return {"response":response}
#     except Exception as e:
#         pass

@app.post("/update_prompt")
async def update_prompt(new_prompt:str):
    try:
        new_prompt_template = new_prompt
        updated_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are a chatbot having a conversation with a human."),  # The persistent system prompt
            MessagesPlaceholder(variable_name="chat_history"),  # Where the memory will be stored.
            HumanMessagePromptTemplate.from_template(new_prompt_template),  # Updated human input template
        ])

        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        chat_llm_chain.prompt = updated_prompt  # Update the prompt template


        llm = ChatOpenAI()

        chat_llm_chain = LLMChain(
            llm=llm,
            verbose=True,
            memory=memory
        )
        # Generate a response using the updated template
        response = chat_llm_chain.predict(human_input="")  # You can pass an empty string or any initial input here
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to update prompt and generate response")



if __name__ == "__main__":
    import uvicorn

    #FastAPI app with Uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
