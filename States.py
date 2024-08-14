from aiogram.fsm.state import StatesGroup, State


class RAGFiles(StatesGroup):
    file = State()


class RAGDeleteState(StatesGroup):
    delete = State()


class RAGState(StatesGroup):
    title = State()
    statement = State()


class RAGQA(StatesGroup):
    question = State()
    answer = State()


class LLMState(StatesGroup):
    get_query = State()
    answer_query = State()


class MenuState(StatesGroup):
    main_menu = State()
