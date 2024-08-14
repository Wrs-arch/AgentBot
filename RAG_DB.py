import os
import configparser

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents.base import Document
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader

config = configparser.ConfigParser()
config.read('config.ini')
config.sections()


def prepare_note_doc(title, statement):
    doc = Document(page_content=title, metadata={'note': statement})
    return doc


def doc_to_chunks(doc):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
    data = text_splitter.split_documents([doc])
    return data


def init_embed_model():
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    return embeddings


def embed_to_db(doc) -> None:
    embeddings = init_embed_model()
    db = FAISS.from_documents(doc, embeddings)

    try:
        new_vector_store = download_db()
        db.merge_from(new_vector_store)

    except Exception as ex:
        print(ex)
        pass

    db.as_retriever()
    db.save_local(config['paths']['faiss_path'])


def download_db():
    new_vector_store = FAISS.load_local(
        config['paths']['faiss_path'], init_embed_model(), allow_dangerous_deserialization=True
    )
    new_vector_store.as_retriever()
    return new_vector_store


def get_db_info():
    database = download_db().docstore.__dict__['_dict']
    paths = database.keys()
    database_keys = [database[key].page_content for key in paths]
    database_values = [database[key].metadata for key in paths]
    return database_keys, database_values

# don't work with pdf/txt files normal
# def load_to_file(database_keys, database_values):
#     my_file = open("D:\Programming\DS_project\\new.txt", "w+")
#     database_values_keys = [list(database_values[number].keys())[0] for number in range(len(database_values))]
#     for number in range(len(database_keys)):
#         my_file.write(database_keys[number] + ' ')
#         my_file.write(database_values[number][database_values_keys[number]] + '\n')
#
#     my_file.close()


def delete_from_files(text):
    database = download_db()
    vec_store = database.docstore.__dict__['_dict']
    query_to_delete = text.split(', ')

    ids = vec_store.keys()
    ids_to_delete = []
    for id in ids:
        if vec_store[id].page_content in query_to_delete:
            ids_to_delete.append(id)
    database.delete(ids_to_delete)
    database.save_local(config['paths']['faiss_path'])


def delete_database():
    os.remove(config['paths']['faiss_index_path'])
    os.remove(config['paths']['faiss_pkl_path'])
    os.rmdir(config['paths']['faiss_path'])


def pdfloader(file_destination):
    loader = PyPDFLoader(file_destination)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    embed_to_db(splits)


def webloader(file_destination):

    loader = WebBaseLoader(file_destination)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    embed_to_db(splits)
