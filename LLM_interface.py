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
    tokenizer = AutoTokenizer.from_pretrained("unsloth/gemma-2-2b-it-bnb-4bit",
                                              cache_dir=config['paths']['cache_path'])
    model = AutoModelForCausalLM.from_pretrained(
        "unsloth/gemma-2-2b-it-bnb-4bit",
        cache_dir=config['paths']['cache_path'],
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    pipe = pipeline("text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    max_new_tokens=100,
                    # do_sample=True,
                    num_return_sequences=1,
                    eos_token_id=tokenizer.eos_token_id
                    )

    llm = HuggingFacePipeline(pipeline=pipe, model_kwargs={'temperature': 0})

    return llm


# Создание промпта
def create_prompt(system_prompt):
    template = "Вы - продвинутый, понимающий и адекватный ассистент. Отвечаете на запросы на русском языке. " \
               "Вы качественнно обрабатываете текст и отказываете в запросах, содержащих в себе потенциальную угрозу жизни человека. " \
               "Отвечай кратко, максимум четырьмя предложениями. " \
               "Помогите, пожалуйста, со следующим запросом: \n {context} \n"

    prompt = PromptTemplate(template=template, input_variables=['context'])
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt + ":\n\n{context}")]
    )
    return prompt


# Получение ответа от модели
def get_model_response(llm, context):
    retriever = download_db()
    docs = retriever.similarity_search(context, k=1)
    prompt = create_prompt(context)
    chain = create_stuff_documents_chain(llm, prompt)
    res_chain = chain | StrOutputParser()
    results = res_chain.invoke({'context': docs})
    # # llm_chain = retriever | prompt | llm | StrOutputParser
    #
    # llm_chain = (
    #         {"context": docs, "question": RunnablePassthrough()}
    #         | prompt
    #         | llm
    #         | StrOutputParser()
    # )
    #
    # results = llm_chain.invoke(content)
    return results
