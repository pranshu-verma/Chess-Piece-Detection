import numpy as np
import cv2
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import ImageGrid


cap = cv2.VideoCapture(0)
# cap.set(3, 1280)
# cap.set(4, 720)

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()


plt.figure(figsize=(10, 10))
plt.xlabel('Webcam image')
plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
plt.show()

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

threshold_image = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

contours, _ = cv2.findContours(threshold_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

contour_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 3)

plt.figure(figsize=(15, 15))

plt.subplot(1, 2, 1)
plt.imshow(threshold_image, cmap='gray')
plt.xlabel('Threshold')

plt.subplot(1, 2, 2)
plt.imshow(contour_image)
plt.xlabel('Contours')

plt.tight_layout()

plt.show()


def find_board_contour(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hierarchy = hierarchy[0]
    children = []

    for i in range(len(contours)):
        epsilon = 0.1 * cv2.arcLength(contours[i], True)
        approx = cv2.approxPolyDP(contours[i], epsilon, True)
        area = cv2.contourArea(approx)
        if len(approx) == 4 and area > 500:
            children.append(hierarchy[i])

    children = np.array(children)
    values, counts = np.unique(children[:, 3], return_counts=True)
    contour = contours[values[np.argmax(counts)]]
    epsilon = 0.1 * cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, epsilon, True)


approx = find_board_contour(frame)

contour_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).copy()
cv2.drawContours(contour_image, [approx], -1, (0, 255, 0), 3)

plt.figure(figsize=(10, 10))
plt.xlabel('Board Contour')
plt.imshow(contour_image)
plt.show()

def correct_perspective(img, approx):
    pts = approx.reshape(4, 2)
    tl, tr, br, bl = (
        pts[np.argmin(np.sum(pts, axis=1))],
        pts[np.argmin(np.diff(pts, axis=1))],
        pts[np.argmax(np.sum(pts, axis=1))],
        pts[np.argmax(np.diff(pts, axis=1))]
    )

    w = max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))
    h = max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))

    src = np.array([tl, tr, br, bl], dtype='float32')
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]],
                   dtype='float32')

    M = cv2.getPerspectiveTransform(src, dst)
    img = cv2.warpPerspective(img, M, (int(w), int(h)))

    return cv2.resize(img, (400, 400))


plt.figure(figsize=(8, 8))

board = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).copy()
board = correct_perspective(board, approx)
plt.imshow(board)
plt.show()


def get_square(img, row, col):
    width = img.shape[0]
    square = width // 8
    x1, y1 = row * square, col * square
    x2, y2 = x1 + square, y1 + square
    return img[x1:x2, y1:y2]


fig = plt.figure(1, (8, 8))
grid = ImageGrid(fig, 111, nrows_ncols=(8, 8), axes_pad=0.1)

# board = dewarp(image, approx)
board = cv2.resize(board, (400, 400))

for i in range(8*8):
    col = i % 8
    row = i // 8
    square = get_square(board, row, col)
    # grid[i].imshow(cv2.cvtColor(square, cv2.COLOR_BGR2RGB))
    grid[i].imshow(square)

print("Showing image")
plt.show()