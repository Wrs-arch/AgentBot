from aiogram import types

main_menu_kb = [[types.KeyboardButton(text='Спросить у модели'), types.KeyboardButton(text='Работа с БД')]]
main_keyboard = types.ReplyKeyboardMarkup(keyboard=main_menu_kb, resize_keyboard=True, one_time_keyboard=True)


db_menu_kb = [[types.KeyboardButton(text='Добавить информацию')],
      # [types.KeyboardButton(text='Вывести содержимое')],
      [types.KeyboardButton(text='Удалить базу данных')],
      [types.KeyboardButton(text='Меню')]]
db_menu_keyboard = types.ReplyKeyboardMarkup(keyboard=db_menu_kb, resize_keyboard=True, one_time_keyboard=True)


yes_no_kb = [[types.KeyboardButton(text='Да')],
          [types.KeyboardButton(text='Нет')]]
yes_no_keyboard = types.ReplyKeyboardMarkup(keyboard=yes_no_kb, resize_keyboard=True, one_time_keyboard=True)


edit_db_menu_kb = [[types.KeyboardButton(text='Создать заметку')],
      # [types.KeyboardButton(text='Записать Вопрос/Ответ')],
      [types.KeyboardButton(text='Загрузить файл')],
      [types.KeyboardButton(text='Меню выбора действий')]]
edit_db_menu_keyboard = types.ReplyKeyboardMarkup(keyboard=edit_db_menu_kb, resize_keyboard=True, one_time_keyboard=True)