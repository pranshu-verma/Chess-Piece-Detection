import cv2
import time
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

from livelossplot import PlotLossesKeras

"""
def find_board_contour(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hierarchy = hierarchy[0]
    children = []

    for i in range(len(contours)):
        epsilon = 0.1 * cv2.arcLength(contours[i],True)
        approx = cv2.approxPolyDP(contours[i], epsilon, True)  
        area = cv2.contourArea(approx)
        if len(approx) == 4 and area > 500:
            children.append(hierarchy[i])

    children = np.array(children)
    values,counts = np.unique(children[:, 3], return_counts=True)
    contour = contours[values[np.argmax(counts)]]
    epsilon = 0.1 * cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, epsilon, True) 


def correct_perspective(img, approx):                                                      
    pts = approx.reshape(4, 2)      
    tl, tr, br, bl =  (
        pts[np.argmin(np.sum(pts, axis=1))],
        pts[np.argmin(np.diff(pts, axis=1))],   
        pts[np.argmax(np.sum(pts, axis=1))],                           
        pts[np.argmax(np.diff(pts, axis=1))]
    )

    w = max(np.linalg.norm(br-bl), np.linalg.norm(tr-tl))
    h = max(np.linalg.norm(tr-br), np.linalg.norm(tl-bl))

    src = np.array([tl, tr, br, bl], dtype='float32')
    dst = np.array([[0, 0],[w, 0],[w, h],[0, h]], dtype='float32')                                        

    M = cv2.getPerspectiveTransform(src, dst)
    img = cv2.warpPerspective(img, M, (int(w), int(h)))

    return cv2.resize(img, (400,400))


def get_square(img, row, col):
    width = img.shape[0]
    square = width // 8
    x1, y1 = row * square, col * square
    x2, y2 = x1 + square, y1 + square
    return img[x1:x2, y1:y2]

train_path = './train_data/images'
label_map = list('KQRBNP_kqrbnp')

capture = cv2.VideoCapture(0)
capture.set(3, 1280)
capture.set(4, 720)

if not capture.isOpened():
    raise RuntimeError('Failed to start camera.')

quit = False
ret, img = capture.read()
approx = find_board_contour(img)

for piece in label_map:
    if quit: break
    path = os.path.join(train_path, piece)
    if not os.path.exists(path):
        os.mkdir(path)
                
    for i in range(8 * 8):
        if quit: break
            
        while True:
            ret, img = capture.read()
            board = correct_perspective(img, approx)
            col = i % 8
            row = i // 8
            square = get_square(board, row, col)
            key_press = cv2.waitKey(5)

            if key_press & 0xFF == ord(' '):
                file_name = '{}_{}.jpg'.format(piece, time.time())
                cv2.imwrite(os.path.join(path, file_name), square)
                break
            
            if key_press & 0xFF == ord('q'):
                quit = True
                break

            width = board.shape[0]
            s = width // 8
            x1, y1 = col * s, row * s
            x2, y2 = x1 + s, y1 + s
            cv2.rectangle(board , (x1, y1), (x2, y2) ,(0, 255, 0), 3)
            cv2.imshow('Capturing piece: {}'.format(piece), board)


cv2.destroyAllWindows()
capture.release()


def rotate(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


images = glob.glob('/train_data/images/*/*.jpg')

for f in images:
    file_name = os.path.basename(f)
    dir_name = os.path.dirname(f)
    img = cv2.imread(f)
    for i in range(1, 4):
        new_img = rotate(img, i * 90)
        new_name = '{}_{}{}'.format(file_name[0],(i * 90),file_name[1:])
        new_path = os.path.join(dir_name, new_name)
        cv2.imwrite(new_path, new_img)

img = cv2.imread('./train_data/images/P/P_1573125920.6008694.jpg')
plt.imshow(img)
plt.show()
"""

img_size = (50, 50)
label_map = list('KQRBNP_kqrbnp')
num_classes = len(label_map)


images = glob.glob('./train_data/images/*/*.jpg')

data_rows = []
for f in images:
    img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, img_size).flatten()
    label = label_map.index(os.path.basename(f)[0])
    data_rows.append(np.insert(img, 0, label))


train_data = pd.DataFrame(data_rows)
print(train_data.head())

train_data.to_csv('./train_data/train_data.csv', index=False)

train_data = pd.read_csv('./train_data/train_data.csv', skiprows=1, header=None)

"""
first_row = train_data.sample(n=1)
label = label_map[first_row[0].item()]
img = first_row.drop([0], axis=1).values
plt.imshow(img.reshape(50,50), cmap='gray')
print('Label: {}'.format(label))
plt.show()
"""

input_shape = (*img_size, 1)
X = train_data.drop([0], axis=1).values
y = to_categorical(train_data[0].values)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = X_train.reshape(X_train.shape[0], *img_size, 1)
X_val = X_val.reshape(X_val.shape[0], *img_size, 1)

X_train = X_train.astype('float32')
X_val = X_val.astype('float32')

X_train /= 255
X_val /= 255

pool_size=(4,4)

model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3),
                 activation='relu',
                 kernel_initializer='he_normal',
                 input_shape=input_shape))

model.add(MaxPooling2D(pool_size))
model.add(Dropout(0.2))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

hist = model.fit(X_train, y_train, batch_size=512, epochs=100, callbacks=[PlotLossesKeras()], verbose=0, validation_data=(X_val, y_val))

print("Accuracy:", hist.history['val_acc'][-1])

model.save('./train_data/model_50x50.hd5')

from keras.models import load_model
model = load_model('/home/chuck/Code/chess/model_50x50.hd5')

def predict(img, model, img_size=(50,50), plot=False):
    img = cv2.resize(img, img_size) 
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY )

    if plot:
        plt.imshow(img, cmap='gray')

    img = img.reshape(1, *img_size, 1) / 255
    pred = model.predict(img)
    return label_map[np.argmax(pred)]


