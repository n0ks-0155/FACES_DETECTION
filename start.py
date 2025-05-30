import numpy as np
import face_recognition


def FindPerson(fname: str, known_faces_encodings: dict):
    test_image = face_recognition.load_image_file(fname)
    test_encodings = face_recognition.face_encodings(test_image)
    titles = list(known_faces_encodings.keys())
    vectors = list(known_faces_encodings.values())
    result_names = []
    for detection in test_encodings:
        result = face_recognition.compare_faces(vectors, detection)
        print(result)
        for i in range(len(result)):
            if result[i]:
                result_names.append(titles[i])
    return result_names


encodings = np.load('face_encodings.npz')
print(encodings['gulev_encoding'])
test_image1 = face_recognition.load_image_file("faces/test/twoperson1.jpeg")

print(FindPerson("faces/test/twoperson1.jpeg", encodings))