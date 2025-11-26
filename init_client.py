from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

deepseek_api = os.environ['DEEPSEEK_API']
deepseek_url = os.environ['DEEPSEEK_URL']
deepseek_model = os.environ['DEEPSEEK_MODEL']

def init_llm(temperature):
    deepseek_client = ChatOpenAI(api_key=deepseek_api,
                                 base_url=deepseek_url,
                                 model=deepseek_model,
                                 temperature=temperature)
    return deepseek_client


