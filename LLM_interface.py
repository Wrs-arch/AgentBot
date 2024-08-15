import torch

from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from RAG_DB import *



# Функция для инициализации модели
def init_llm():
    config = configparser.ConfigParser()
    config.read('config.ini')
    config.sections()
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it",
                                              cache_dir=config['paths']['cache_path'])
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-2b-it",
        cache_dir=config['paths']['cache_path'],
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    pipe = pipeline("text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    max_new_tokens=200,
                    # do_sample=True,
                    num_return_sequences=1,
                    eos_token_id=tokenizer.eos_token_id
                    )

    llm = HuggingFacePipeline(pipeline=pipe, model_kwargs={'temperature': 0})

    return llm


# Создание промпта
def create_prompt():
    template = """Вы - продвинутый, понимающий и адекватный ассистент. Отвечаете на запросы на русском языке.\n
               Вы качественнно обрабатываете текст и отказываете в запросах, содержащих в себе потенциальную угрозу жизни человека.\n
               Отвечай полноценно, качественно, максимум 10 предложений.\n
               Помогите, пожалуйста, со следующим запросом: \n {query} \n 
               Ответ ищи, основываясь на тексте, но не ограничивайся им: {context}
               Ответ: """

    prompt = PromptTemplate(template=template, input_variables=['query', 'context'])
    return prompt


# Получение ответа от модели
def get_model_response(llm, query):
    retriever = download_db()
    docs = retriever.similarity_search(query, k=2)
    prompt = create_prompt()
    chain = create_stuff_documents_chain(llm, prompt)
    res_chain = chain | StrOutputParser()
    results = res_chain.invoke({'query': query, 'context': docs})
    return results
