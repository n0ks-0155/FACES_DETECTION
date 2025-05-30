import logging
import io
from typing import List, Tuple

import face_recognition
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from telegram import Update, InputFile
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# Настройка логирования
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

TOKEN = "7937525572:AAEUh7eoNsVQX_vK3RMSkOqFBnt_DjexXV4"

known_faces_db = np.load('face_encodings.npz')


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /start"""
    await update.message.reply_text(
        "👋 Привет! Я бот для распознавания лиц.\n\n"
        "Отправьте мне фото, и я попробую распознать лица на нем.\n\n"
        "Доступные команды:\n"
        "/add_face [имя] - добавить лицо в базу\n"
        "/list_faces - список известных лиц\n"
        "/help - справка"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /help"""
    await update.message.reply_text(
        "ℹ️ Помощь по боту:\n\n"
        "1. Просто отправьте фото - я попробую распознать лица\n"
        "2. Чтобы добавить новое лицо, отправьте фото с командой /add_face [имя]\n"
        "3. /list_faces - показать все известные лица"
    )


async def list_faces(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /list_faces"""
    if not known_faces_db:
        await update.message.reply_text("ℹ️ В базе нет известных лиц.")
        return

    faces_list = "\n".join([f"• {name}" for name in known_faces_db.keys()])
    await update.message.reply_text(f"📋 Известные лица:\n\n{faces_list}")


async def add_face(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Добавление нового лица в базу"""


    print('-------------------------')
    if not context.args:
        await update.message.reply_text("❌ Укажите имя: /add_face [имя]")
        return
    logger.debug('Enter add face')
    name = " ".join(context.args)

    if not update.message.photo:
        await update.message.reply_text("❌ Нужно отправить фото!")
        return

    # Скачиваем фото
    photo_file = await update.message.photo[-1].get_file()
    photo_bytes = io.BytesIO()
    await photo_file.download_to_memory(out=photo_bytes)
    photo_bytes.seek(0)

    # Ищем лица на фото
    image = face_recognition.load_image_file(photo_bytes)
    encodings = face_recognition.face_encodings(image,model="large")

    if not encodings:
        await update.message.reply_text("❌ Не найдено лиц на фото!")
        return

    # Сохраняем первое найденное лицо
    known_faces_db[name] = encodings[0]
    await update.message.reply_text(f"✅ Лицо '{name}' добавлено в базу!")


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработка присланных фото"""
    # Скачиваем фото
    print('-------------------------')

    photo_file = await update.message.photo[-1].get_file()
    photo_bytes = io.BytesIO()
    await photo_file.download_to_memory(out=photo_bytes)
    photo_bytes.seek(0)

    # Загружаем изображение
    image = face_recognition.load_image_file(photo_bytes)

    # Находим лица
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    # Загружаем известные лица
    known_faces = [
        (name, np.frombuffer(encoding, dtype=np.float64))
        for name, encoding in known_faces_db.items()
    ]

    # Распознаем лица
    recognized = []
    for face_encoding in face_encodings:
        name = "Неизвестный"
        if known_faces:
            distances = face_recognition.face_distance(
                [enc for _, enc in known_faces],
                face_encoding
            )
            min_index = np.argmin(distances)
            if distances[min_index] < 0.6:  # Порог распознавания
                name = known_faces[min_index][0]
        recognized.append(name)

    # Рисуем рамки
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)

    for (top, right, bottom, left), name in zip(face_locations, recognized):
        # Рамка вокруг лица
        draw.rectangle(((left, top), (right, bottom)), outline="red", width=2)

        # Подпись
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    for (top, right, bottom, left), name in zip(face_locations, recognized):
        draw.rectangle(((left, top), (right, bottom)), outline="red", width=2)

        bbox = draw.textbbox((0, 0), name, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        draw.rectangle(
            ((left, bottom - text_height - 10), (left + text_width + 10, bottom)),
            fill="red"
        )
        draw.text(
            (left + 5, bottom - text_height - 5),
            name,
            fill="white",
            font=font
        )

    # Сохраняем результат
    output = io.BytesIO()
    pil_image.save(output, format="JPEG")
    output.seek(0)

    # Отправляем результат
    await update.message.reply_photo(
        photo=InputFile(output),
        caption=f"Найдено лиц: {len(recognized)}"
    )


def main() -> None:
    """Запуск бота"""
    # Создаем Application
    application = Application.builder().token(TOKEN).build()

    # Регистрируем обработчики
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("list_faces", list_faces))
    application.add_handler(CommandHandler("add_face", add_face))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    # Запускаем бота
    application.run_polling()


if __name__ == "__main__":
    main()