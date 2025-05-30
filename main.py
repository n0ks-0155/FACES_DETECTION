import sys

import face_recognition
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from main2 import show_cropped_images
#
img_zavalin ="faces/Zavalin2.png"
img_zimnuhov = "faces/Zimnuhov3.jpeg"
img_svista = "faces/Svista20.jpeg"
img_russkikh = "faces/Russkikh.jpeg"
img_sarapulov = "faces/Sarapulov.jpeg"
img_gulev = "faces/Gulev10.jpeg"
img_matveev = "faces/Matveev10.jpg"
img_karavanov = "faces/Karavanov10.jpeg"
img_zinov = "faces/zinov.jpeg"
#
# image_path = img_matveev
#
# image = face_recognition.load_image_file(image_path)
# face_locations = face_recognition.face_locations(image)
#
#
# img = cv.imread(image_path)
# img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
#
# plt.imshow(img)
# plt.axis('off')
#
# for face_location in face_locations:
#     plt.plot(face_location[3], face_location[0], 'ro')
#     plt.plot(face_location[1], face_location[0], 'r+')
#     plt.plot(face_location[3], face_location[2], 'bo')
#     plt.plot(face_location[1], face_location[2], 'b+')
#
# plt.show()
#
# for face_location in face_locations:
#     x1, y1 = face_location[3], face_location[2]
#     x2, y2 = face_location[1], face_location[2]
#     x3, y3 = face_location[1], face_location[0]
#     x4, y4 = face_location[3], face_location[0]
#
# top_left_x = min([x1,x2,x3,x4])
# top_left_y = min([y1,y2,y3,y4])
# bot_right_x = max([x1,x2,x3,x4])
# bot_right_y = max([y1,y2,y3,y4])
#
# cropped_image = img[top_left_y:bot_right_y, top_left_x:bot_right_x]
# plt.imshow(cropped_image)
# plt.axis('off')
# plt.show()
# cv.imwrite('faces/cut/matveev.jpg', cv.cvtColor(cropped_image, cv.COLOR_RGB2BGR))



#
# img_zavalin ="faces/cut/zavalin.jpg"
# img_zimnuhov = "faces/cut/zimnuhov.jpg"
# img_svista = "faces/cut/svista.jpg"
# img_russkikh = "faces/cut/russkikh.jpg"
# img_sarapulov = "faces/cut/sarapulov.jpg"
# img_gulev = "faces/cut/gulev.jpg"
# img_matveev = "faces/cut/matveev.jpg"
# img_karavanov = "faces/cut/karavanov.jpg"
# img_zinov = "faces/cut/zinov.jpg"


# encodings = dict()
# zinov_train = face_recognition.load_image_file(img_zinov)
# encodings['zinov_encoding'] = face_recognition.face_encodings(zinov_train)[0]
# karavanov_train = face_recognition.load_image_file(img_karavanov)
# encodings['karavanov_encoding'] = face_recognition.face_encodings(karavanov_train)[0]
# matveev_train = face_recognition.load_image_file(img_matveev)
# encodings['matveev_encoding'] = face_recognition.face_encodings(matveev_train)[0]
# gulev_train = face_recognition.load_image_file(img_gulev)
# encodings['gulev_encoding'] = face_recognition.face_encodings(gulev_train)[0]
# sarapulov_train = face_recognition.load_image_file(img_sarapulov)
# encodings['sarapulov_encoding'] = face_recognition.face_encodings(sarapulov_train)[0]
# russkikh_train = face_recognition.load_image_file(img_russkikh)
# encodings['russkikh_encoding'] = face_recognition.face_encodings(russkikh_train)[0]
# svista_train = face_recognition.load_image_file(img_svista)
# encodings['svista_encoding'] = face_recognition.face_encodings(svista_train)[0]
# zimnuhov_train = face_recognition.load_image_file(img_zimnuhov)
# encodings['zimnuhov_encoding'] = face_recognition.face_encodings(zimnuhov_train)[0]
# zavalin_train = face_recognition.load_image_file(img_zavalin)
# encodings['zavalin_encoding'] = face_recognition.face_encodings(zavalin_train)[0]
#
# np.savez('face_encodings.npz',**encodings)
# sys.exit(0)

encodings = np.load('face_encodings.npz')

print(list(encodings.keys()))
sys.exit(0)
#show_cropped_images()

test_image1 = face_recognition.load_image_file("faces/test/twoperson1.jpeg")
test_image2 = face_recognition.load_image_file("faces/test/twoperson.jpeg")
test_image3 = face_recognition.load_image_file("faces/test/allperson2.jpeg")
test_image4 = face_recognition.load_image_file("faces/test/allperson.jpeg")
test_image5 = face_recognition.load_image_file("faces/test/zinovperson.png")
test_image6 = face_recognition.load_image_file("faces/test/gulevperson.png")
test_image7 = face_recognition.load_image_file("faces/test/unknown_02.jpg")

test1_encoding = face_recognition.face_encodings(test_image1)
test2_encoding = face_recognition.face_encodings(test_image2)
test3_encoding = face_recognition.face_encodings(test_image3)
test4_encoding = face_recognition.face_encodings(test_image4)
test5_encoding = face_recognition.face_encodings(test_image5)
test6_encoding = face_recognition.face_encodings(test_image6)
test7_encoding = face_recognition.face_encodings(test_image7)

trained_images = [zinov_encoding, karavanov_encoding, matveev_encoding, gulev_encoding, sarapulov_encoding, russkikh_encoding, svista_encoding, zimnuhov_encoding, zavalin_encoding]
trained_faces = np.array(["zinov", "karavanov", "matveev", "gulev", "sarapulov", "russkikh", "svista", "zimnuhov", "zavalin"])

test4_results = []

for detection in test4_encoding:
    result = face_recognition.compare_faces(trained_images, detection)
    test4_results.append(trained_faces[result])

print(test4_results)


test_img4 = plt.imread('faces/test/allperson.jpeg')
plt.title('detected faces: \n' + str(test4_results), fontsize='small')
plt.axis('off')
plt.imshow(test_img4)
plt.show()



test1_results = []

for detection in test1_encoding:
    result = face_recognition.compare_faces(trained_images, detection)
    test1_results.append(trained_faces[result])

print(test1_results)

test_img1 = plt.imread('faces/test/allperson2.jpeg')
plt.title('detected faces: \n' + str(test1_results), fontsize='small')
plt.axis('off')
plt.imshow(test_img1)
plt.show()

test2_results = []

for detection in test2_encoding:
    result = face_recognition.compare_faces(trained_images, detection)
    test2_results.append(trained_faces[result])

print(test2_results)

test_img2 = plt.imread('faces/test/twoperson1.jpeg')
plt.title('detected faces: \n' + str(test2_results), fontsize='small')
plt.axis('off')
plt.imshow(test_img2)
plt.show()

test3_results = []

for detection in test3_encoding:
    result = face_recognition.compare_faces(trained_images, detection)
    test3_results.append(trained_faces[result])

print(test3_results)

test_img3 = plt.imread('faces/test/zinovperson.png')
plt.title('detected faces: \n' + str(test3_results), fontsize='small')
plt.axis('off')
plt.imshow(test_img3)
plt.show()

sys.exit(0)
test5_results = []

for detection in test5_encoding:
    result = face_recognition.compare_faces(trained_images, detection)
    test5_results.append(trained_faces[result])

test_img5 = plt.imread('faces/test/gulevperson.png')
plt.title('detected faces: \n' + str(test5_results), fontsize='small')
plt.axis('off')
plt.imshow(test_img5)
plt.show()

test6_results = []

for detection in test6_encoding:
    result = face_recognition.compare_faces(trained_images, detection)
    test6_results.append(trained_faces[result])

test_img6 = plt.imread('faces/test/twoperson.jpeg')
plt.title('detected faces: \n' + str(test6_results), fontsize='small')
plt.axis('off')
plt.imshow(test_img6)
plt.show()

test7_results = []

for detection in test7_encoding:
    result = face_recognition.compare_faces(trained_images, detection)
    test7_results.append(trained_faces[result])

test_img7 = plt.imread('faces/test/unknown_02.jpg')
plt.title('detected faces: \n' + str(test7_results), fontsize='small')
plt.axis('off')
plt.imshow(test_img7)
plt.show()

from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Загружаем изображение
unknown_image = face_recognition.load_image_file('faces/test/4Kallperson.jpeg')

# Находим все лица и их кодировки
face_locations = face_recognition.face_locations(unknown_image)
face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

# Создаем PIL изображение для рисования
pil_image = Image.fromarray(unknown_image)
draw = ImageDraw.Draw(pil_image)

# Шрифт для подписей
try:
    font = ImageFont.truetype("arial.ttf", 20)
except:
    font = ImageFont.load_default()

# Список имен для распознанных лиц
image_labels = ["gulev", "russkikh", "zinov", "lysak", "matveev",
                "sarapulov", "svista", "zimnuhov", "zavalin"]

for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    # Сравниваем с известными лицами
    matches = face_recognition.compare_faces(face_encodings, face_encoding)
    name = "Unknown"

    # Ищем лучшее совпадение
    face_distances = face_recognition.face_distance(face_encodings, face_encoding)
    best_match_index = np.argmin(face_distances)

    if matches[best_match_index]:
        name = image_labels[best_match_index]

    # Рисуем прямоугольник вокруг лица
    draw.rectangle(((left, top), (right, bottom)), outline=(255, 0, 0), width=2)

    # Получаем размеры текста (новый способ)
    bbox = draw.textbbox((0, 0), name, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Рисуем подложку для текста
    draw.rectangle(
        ((left, bottom - text_height - 10), (right, bottom)),
        fill=(255, 0, 0),
        outline=(255, 0, 0)
    )

    # Рисуем текст
    draw.text(
        (left + 6, bottom - text_height - 5),
        name,
        fill=(255, 255, 255),
        font=font
    )

# Показываем результат
pil_image.show()

# Загружаем изображение
unknown_image2 = face_recognition.load_image_file('faces/test/twoperson1.jpeg')

# Находим все лица и их кодировки
face_locations2 = face_recognition.face_locations(unknown_image2)
face_encodings2 = face_recognition.face_encodings(unknown_image2, face_locations2)

# Создаем PIL изображение для рисования
pil_image2 = Image.fromarray(unknown_image2)
draw = ImageDraw.Draw(pil_image2)

# Шрифт для подписей
try:
    font = ImageFont.truetype("arial.ttf", 20)
except:
    font = ImageFont.load_default()

# Список имен для распознанных лиц
image_labels = ["gulev", "russkikh", "zinov", "lysak", "matveev",
                "sarapulov", "svista", "zimnuhov", "zavalin"]

for (top, right, bottom, left), face_encoding in zip(face_locations2, face_encodings2):
    # Сравниваем с известными лицами
    matches = face_recognition.compare_faces(face_encodings, face_encoding)
    name = "Unknown"

    # Ищем лучшее совпадение
    face_distances = face_recognition.face_distance(face_encodings, face_encoding)
    best_match_index = np.argmin(face_distances)

    if matches[best_match_index]:
        name = image_labels[best_match_index]

    # Рисуем прямоугольник вокруг лица
    draw.rectangle(((left, top), (right, bottom)), outline=(255, 0, 0), width=2)

    # Получаем размеры текста (новый способ)
    bbox = draw.textbbox((0, 0), name, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Рисуем подложку для текста
    draw.rectangle(
        ((left, bottom - text_height - 10), (right, bottom)),
        fill=(255, 0, 0),
        outline=(255, 0, 0)
    )

    # Рисуем текст
    draw.text(
        (left + 6, bottom - text_height - 5),
        name,
        fill=(255, 255, 255),
        font=font
    )

# Показываем результат
pil_image2.show()

# Загружаем изображение
unknown_image3 = face_recognition.load_image_file('faces/test/zinovperson.png')

# Находим все лица и их кодировки
face_locations3 = face_recognition.face_locations(unknown_image3)
face_encodings3 = face_recognition.face_encodings(unknown_image3, face_locations3)

# Создаем PIL изображение для рисования
pil_image3 = Image.fromarray(unknown_image3)
draw = ImageDraw.Draw(pil_image3)

# Шрифт для подписей
try:
    font = ImageFont.truetype("arial.ttf", 20)
except:
    font = ImageFont.load_default()

# Список имен для распознанных лиц
image_labels = ["gulev", "russkikh", "zinov", "lysak", "matveev",
                "sarapulov", "svista", "zimnuhov", "zavalin"]

for (top, right, bottom, left), face_encoding in zip(face_locations3, face_encodings3):
    # Сравниваем с известными лицами
    matches = face_recognition.compare_faces(face_encodings, face_encoding)
    name = "Unknown"

    # Ищем лучшее совпадение
    face_distances = face_recognition.face_distance(face_encodings, face_encoding)
    best_match_index = np.argmin(face_distances)

    if matches[best_match_index]:
        name = image_labels[best_match_index]

    # Рисуем прямоугольник вокруг лица
    draw.rectangle(((left, top), (right, bottom)), outline=(255, 0, 0), width=2)

    # Получаем размеры текста (новый способ)
    bbox = draw.textbbox((0, 0), name, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Рисуем подложку для текста
    draw.rectangle(
        ((left, bottom - text_height - 10), (right, bottom)),
        fill=(255, 0, 0),
        outline=(255, 0, 0)
    )

    # Рисуем текст
    draw.text(
        (left + 6, bottom - text_height - 5),
        name,
        fill=(255, 255, 255),
        font=font
    )

# Показываем результат
pil_image3.show()

# Загружаем изображение
unknown_image4 = face_recognition.load_image_file('faces/test/allperson.jpeg')

# Находим все лица и их кодировки
face_locations4 = face_recognition.face_locations(unknown_image4)
face_encodings4 = face_recognition.face_encodings(unknown_image4, face_locations4)

# Создаем PIL изображение для рисования
pil_image4 = Image.fromarray(unknown_image4)
draw = ImageDraw.Draw(pil_image4)

# Шрифт для подписей
try:
    font = ImageFont.truetype("arial.ttf", 20)
except:
    font = ImageFont.load_default()

# Список имен для распознанных лиц
image_labels = ["gulev", "russkikh", "zinov", "lysak", "matveev",
                "sarapulov", "svista", "zimnuhov", "zavalin"]

for (top, right, bottom, left), face_encoding in zip(face_locations4, face_encodings4):
    # Сравниваем с известными лицами
    matches = face_recognition.compare_faces(face_encodings, face_encoding)
    name = "Unknown"

    # Ищем лучшее совпадение
    face_distances = face_recognition.face_distance(face_encodings, face_encoding)
    best_match_index = np.argmin(face_distances)

    if matches[best_match_index]:
        name = image_labels[best_match_index]

    # Рисуем прямоугольник вокруг лица
    draw.rectangle(((left, top), (right, bottom)), outline=(255, 0, 0), width=2)

    # Получаем размеры текста (новый способ)
    bbox = draw.textbbox((0, 0), name, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Рисуем подложку для текста
    draw.rectangle(
        ((left, bottom - text_height - 10), (right, bottom)),
        fill=(255, 0, 0),
        outline=(255, 0, 0)
    )

    # Рисуем текст
    draw.text(
        (left + 6, bottom - text_height - 5),
        name,
        fill=(255, 255, 255),
        font=font
    )

# Показываем результат
pil_image4.show()

# Загружаем изображение
unknown_image5 = face_recognition.load_image_file('faces/test/gulevperson.png')

# Находим все лица и их кодировки
face_locations5 = face_recognition.face_locations(unknown_image5)
face_encodings5 = face_recognition.face_encodings(unknown_image5, face_locations5)

# Создаем PIL изображение для рисования
pil_image5 = Image.fromarray(unknown_image5)
draw = ImageDraw.Draw(pil_image5)

# Шрифт для подписей
try:
    font = ImageFont.truetype("arial.ttf", 20)
except:
    font = ImageFont.load_default()

# Список имен для распознанных лиц
image_labels = ["gulev", "russkikh", "zinov", "lysak", "matveev",
                "sarapulov", "svista", "zimnuhov", "zavalin"]

for (top, right, bottom, left), face_encoding in zip(face_locations5, face_encodings5):
    # Сравниваем с известными лицами
    matches = face_recognition.compare_faces(face_encodings, face_encoding)
    name = "Unknown"

    # Ищем лучшее совпадение
    face_distances = face_recognition.face_distance(face_encodings, face_encoding)
    best_match_index = np.argmin(face_distances)

    if matches[best_match_index]:
        name = image_labels[best_match_index]

    # Рисуем прямоугольник вокруг лица
    draw.rectangle(((left, top), (right, bottom)), outline=(255, 0, 0), width=2)

    # Получаем размеры текста (новый способ)
    bbox = draw.textbbox((0, 0), name, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Рисуем подложку для текста
    draw.rectangle(
        ((left, bottom - text_height - 10), (right, bottom)),
        fill=(255, 0, 0),
        outline=(255, 0, 0)
    )

    # Рисуем текст
    draw.text(
        (left + 6, bottom - text_height - 5),
        name,
        fill=(255, 255, 255),
        font=font
    )

# Показываем результат
pil_image5.show()

# Загружаем изображение
unknown_image6 = face_recognition.load_image_file('faces/test/twoperson.jpeg')

# Находим все лица и их кодировки
face_locations6 = face_recognition.face_locations(unknown_image6)
face_encodings6 = face_recognition.face_encodings(unknown_image6, face_locations6)

# Создаем PIL изображение для рисования
pil_image6 = Image.fromarray(unknown_image6)
draw = ImageDraw.Draw(pil_image6)

# Шрифт для подписей
try:
    font = ImageFont.truetype("arial.ttf", 20)
except:
    font = ImageFont.load_default()

# Список имен для распознанных лиц
image_labels = ["gulev", "russkikh", "zinov", "lysak", "matveev",
                "sarapulov", "svista", "zimnuhov", "zavalin"]

for (top, right, bottom, left), face_encoding in zip(face_locations6, face_encodings6):
    # Сравниваем с известными лицами
    matches = face_recognition.compare_faces(face_encodings, face_encoding)
    name = "Unknown"

    # Ищем лучшее совпадение
    face_distances = face_recognition.face_distance(face_encodings, face_encoding)
    best_match_index = np.argmin(face_distances)

    if matches[best_match_index]:
        name = image_labels[best_match_index]

    # Рисуем прямоугольник вокруг лица
    draw.rectangle(((left, top), (right, bottom)), outline=(255, 0, 0), width=2)

    # Получаем размеры текста (новый способ)
    bbox = draw.textbbox((0, 0), name, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Рисуем подложку для текста
    draw.rectangle(
        ((left, bottom - text_height - 10), (right, bottom)),
        fill=(255, 0, 0),
        outline=(255, 0, 0)
    )

    # Рисуем текст
    draw.text(
        (left + 6, bottom - text_height - 5),
        name,
        fill=(255, 255, 255),
        font=font
    )

# Показываем результат
pil_image6.show()

# Загружаем изображение
unknown_image7 = face_recognition.load_image_file('faces/test/unknown_02.jpg')

# Находим все лица и их кодировки
face_locations7 = face_recognition.face_locations(unknown_image7)
face_encodings7 = face_recognition.face_encodings(unknown_image7, face_locations7)

# Создаем PIL изображение для рисования
pil_image7 = Image.fromarray(unknown_image7)
draw = ImageDraw.Draw(pil_image7)

# Шрифт для подписей
try:
    font = ImageFont.truetype("arial.ttf", 20)
except:
    font = ImageFont.load_default()

# Список имен для распознанных лиц
image_labels = ["gulev", "russkikh", "zinov", "lysak", "matveev",
                "sarapulov", "svista", "zimnuhov", "zavalin"]

for (top, right, bottom, left), face_encoding in zip(face_locations7, face_encodings7):
    # Сравниваем с известными лицами
    matches = face_recognition.compare_faces(face_encodings, face_encoding)
    name = "Unknown"

    # Ищем лучшее совпадение
    face_distances = face_recognition.face_distance(face_encodings, face_encoding)
    best_match_index = np.argmin(face_distances)

    if matches[best_match_index]:
        name = image_labels[best_match_index]

    # Рисуем прямоугольник вокруг лица
    draw.rectangle(((left, top), (right, bottom)), outline=(255, 0, 0), width=2)

    # Получаем размеры текста (новый способ)
    bbox = draw.textbbox((0, 0), name, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Рисуем подложку для текста
    draw.rectangle(
        ((left, bottom - text_height - 10), (right, bottom)),
        fill=(255, 0, 0),
        outline=(255, 0, 0)
    )

    # Рисуем текст
    draw.text(
        (left + 6, bottom - text_height - 5),
        name,
        fill=(255, 255, 255),
        font=font
    )

# Показываем результат
pil_image7.show()

with open('features/face_encodings.npy', 'wb') as f:
    np.save(f, face_encodings)

with open('features/image_labels.npy', 'wb') as f:
    np.save(f, image_labels)

with open('features/face_encodings.npy', 'rb') as f:
    feature_vectors = np.load(f)

with open('features/image_labels.npy', 'rb') as f:
    feature_labels = np.load(f)

np.unique(feature_labels, return_counts=True)

feature_vectors.shape