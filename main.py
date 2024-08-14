import asyncio
import configparser

from aiogram import Bot, types, Dispatcher
from aiogram import F
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext

from keyboards import *
from States import *
from RAG_DB import *
from LLM_interface import *

dp = Dispatcher()
# Старт и создание меню


@dp.message(Command('start'))
async def start(message: types.Message) -> None:
    await message.answer('Welcome!', reply_markup=main_keyboard)


@dp.message(F.text == 'Меню')
@dp.message(MenuState.main_menu)
async def menu(message: types.Message, state: FSMContext) -> None:
    await message.answer('Главное меню', reply_markup=main_keyboard)
    await state.set_state(None)


@dp.message(F.text == 'Меню выбора действий')
async def menu_actions(message: types.Message, state: FSMContext) -> None:
    await message.answer('Выберите, что вы хотите сделать', reply_markup=db_menu_keyboard)
    await state.set_state(None)


# Обработка запросов с базой данных

@dp.message(F.text == 'Работа с БД')
async def db_menu(message: types.Message) -> None:
    await message.answer('Выберите, что вы хотите сделать', reply_markup=db_menu_keyboard)


@dp.message(F.text == 'Удалить базу данных')
async def start_remove_info(message: types.Message, state: FSMContext) -> None:
    await message.answer('Вы уверены?', reply_markup=yes_no_keyboard)
    await state.set_state(RAGDeleteState.delete)


@dp.message(F.text == 'Загрузить файл')
async def load_file(message: types.Message, state: FSMContext) -> None:
    await message.answer('На данный момент поддерживается закачка: pdf, url')
    await state.set_state(RAGFiles.file)


# @dp.message(F.text == 'Вывести содержимое')
# async def get_info(message: types.Message) -> None:
#     database_keys, database_values = get_DB_info()
#     load_to_file(database_keys, database_values)
#     file = FSInputFile("new.txt")
#     await message.answer_document(file)


@dp.message(RAGFiles.file)
async def load_pdf(message: types.Message, state: FSMContext) -> None:
    if message.document:
        file_id = message.document.file_id
        file = await bot.get_file(file_id)
        file_path = file.file_path
        file_destination = "D:\Programming\DS_project\downloaders\\text.pdf"
        await bot.download_file(file_path, file_destination)
        pdfloader(file_destination)
    else:
        url = message.text
        webloader(url)
        await state.set_state(None)


@dp.message(RAGDeleteState.delete)
async def remove_info(message: types.Message, state: FSMContext) -> None:
    msg = message.text
    if msg == 'Да':
        delete_database()
        await state.set_state(None)
        await message.answer('База данных удалена. \nВы возвращены в главное меню.', reply_markup=main_keyboard)
    elif msg == 'Нет':
        await state.set_state(None)
        await message.answer('База данных не удалена \nВы возвращены в главное меню.', reply_markup=main_keyboard)


@dp.message(F.text == 'Добавить информацию')
async def add_info(message: types.Message) -> None:
    await message.answer('Выберите, что вы хотите внести', reply_markup=edit_db_menu_keyboard)


@dp.message(F.text == 'Создать заметку')
async def create_note(message: types.Message, state: FSMContext) -> None:
    await message.answer('Опишите название заметки')
    await state.set_state(RAGState.title)


@dp.message(RAGState.title)
async def title_note(message: types.Message, state: FSMContext) -> None:
    global title_str
    title_str = message.text
    await message.answer('Опишите содержание заметку')
    await state.set_state(RAGState.statement)


@dp.message(RAGState.statement)
async def statement_note(message: types.Message, state: FSMContext) -> None:
    statement = message.text
    dic = prepare_note_doc(title_str, statement)
    data = doc_to_chunks(dic)
    embed_to_db(data)
    await state.set_state(None)

# Обработка LLM запросов


@dp.message(F.text == 'Спросить у модели')
async def request_to_LLM(message: types.Message, state: FSMContext) -> None:
    await message.answer('Напишите ваш запрос')
    await state.set_state(LLMState.get_query)


@dp.message(LLMState.get_query)
async def get_query_LLM(message: types.Message, state: FSMContext) -> None:
    database = download_db()

    llm_query = message.text
    llm_answer = get_model_response(llm, llm_query)
    await message.answer(llm_answer)
    await state.set_state(None)


# Функция main и запуск

async def main() -> None:
    global bot
    config.read('config.ini')
    config.sections()
    token = config['DEFAULT']['API_TOKEN']
    bot = Bot(token)
    await dp.start_polling(bot)


if __name__ == "__main__":
    llm = init_llm()
    asyncio.run(main())
