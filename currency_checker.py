import cv2
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
from io import BytesIO
import os

def check_color(reference_image, captured_image, threshold=30):
    try:
        captured_hsv = cv2.cvtColor(captured_image, cv2.COLOR_BGR2HSV)
        reference_hsv = cv2.cvtColor(reference_image, cv2.COLOR_BGR2HSV)
    except cv2.error as e:
        st.error(f"Error converting image to HSV: {e}")
        return 1  # Assume fake if conversion fails

    captured_hsv[:, :, 2] = cv2.equalizeHist(captured_hsv[:, :, 2])
    reference_hsv[:, :, 2] = cv2.equalizeHist(reference_hsv[:, :, 2])

    x, y, width, height = 1240, 0, 150, 800

    captured_hsv = captured_hsv[y:y + height, x:x + width]
    reference_hsv = reference_hsv[y:y + height, x:x + width]

    captured_mean_saturation = np.mean(captured_hsv[:, :, 1])
    reference_mean_saturation = np.mean(reference_hsv[:, :, 1])

    saturation_difference = abs(reference_mean_saturation - captured_mean_saturation)

    if saturation_difference > threshold:
        st.write("Currency image has a different color illumination than the original.")
        return 1
    else:
        st.write("Currency image has a similar color illumination as the original.")
        return 0

def check_bleedlines(reference_left_lines, current_left_lines):
    fig, axes = plt.subplots(3, 2, figsize=(7, 7))

    fake = 0

    for i in range(3):
        reference_image = reference_left_lines[i]
        current_image = current_left_lines[i]

        axes[i, 0].imshow(cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB), cmap='gray')
        axes[i, 0].set_title("Reference Image")

        axes[i, 1].imshow(cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB), cmap='gray')
        axes[i, 1].set_title("Current Image")

        reference_width = reference_image.shape[1]
        current_width = current_image.shape[1]
        st.write(f"Real blend_line {i + 1}: {reference_width} -------- Current blend_line {i + 1}: {current_width}")

        if abs(reference_width - current_width) > 2:
            fake = 1

    if fake:
        st.write("The currency is Fake")
    else:
        st.write("The currency is Real")

    plt.tight_layout()
    st.pyplot(fig)

    return fake

def extract_features(current_image):
    gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 2)

    lower_threshold = 160
    upper_threshold = 240

    edges = cv2.Canny(blurred, lower_threshold, upper_threshold)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(closed)

    for contour in contours:
        cv2.drawContours(filled, [contour], 0, (255, 255, 255), cv2.FILLED)

    image_copy = current_image.copy()
    left_lines = []
    bottom_serial = []
    top_serial = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(current_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        region = image_copy[y:y + h, x:x + w]

        if x < 5:
            left_lines.append(region)
        if 820 < x < 1000 and y > 400:
            bottom_serial.append(region)
        if y < 200 and 160 < x < 220:
            top_serial.append(region)

    return left_lines, bottom_serial, top_serial

def check_font(current_bottom_serial):
    flag = 0
    fig, axes = plt.subplots(3, 2, figsize=(7, 7))

    real_numbers = [21, 22, 24, 27, 29, 32]
    numbers = [3, 3, 6, 0, 4, 9]
    image = current_bottom_serial

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 2)

    lower_threshold = 100
    upper_threshold = 250

    edges = cv2.Canny(blurred, lower_threshold, upper_threshold)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(closed)

    for contour in contours:
        cv2.drawContours(filled, [contour], 0, (255, 255, 255), cv2.FILLED)
    image_copy = image.copy()
    all_numbers = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        region = image_copy[y:y + h, x:x + w]
        if x < 4:
            all_numbers.insert(0, region)
        else:
            all_numbers.append(region)

    for i in range(6):
        if abs(all_numbers[i].shape[1] - real_numbers[i]) > 2:
            flag = 1

    i = 0
    for j in range(3):
        axes[j, 0].imshow(cv2.cvtColor(all_numbers[i], cv2.COLOR_BGR2RGB), cmap='gray')
        st.write(f"size of {numbers[i]} -----> {all_numbers[i].shape[1]} ----- real:{real_numbers[i]}")
        axes[j, 1].imshow(cv2.cvtColor(all_numbers[i + 1], cv2.COLOR_BGR2RGB), cmap='gray')
        st.write(f"size of {numbers[i + 1]} -----> {all_numbers[i + 1].shape[1]} ----- real:{real_numbers[i + 1]}")
        i += 2

    if flag:
        st.write("fontsize check: Fake")
    else:
        st.write("fontsize check: Real")

    plt.tight_layout()
    st.pyplot(fig)

    return flag

def check_serial(reference_bottom_serial, reference_top_serial, current_bottom_serial, current_top_serial, threshold=5):
    fake = 0
    st.write("Serial_check:-")
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))

    current_bottom_serial = cv2.cvtColor(current_bottom_serial, cv2.COLOR_BGR2GRAY)
    current_top_serial = cv2.cvtColor(current_top_serial, cv2.COLOR_BGR2GRAY)

    _, current_bottom_serial = cv2.threshold(current_bottom_serial, 170, 255, cv2.THRESH_BINARY)
    _, current_top_serial = cv2.threshold(current_top_serial, 120, 255, cv2.THRESH_BINARY)

    current_bottom_serial = cv2.bitwise_not(current_bottom_serial)
    current_top_serial = cv2.bitwise_not(current_top_serial)

    axes[0].imshow(cv2.cvtColor(current_bottom_serial, cv2.COLOR_BGR2RGB), cmap='gray')
    axes[0].set_title("Bottom serial")

    axes[1].imshow(cv2.cvtColor(current_top_serial, cv2.COLOR_BGR2RGB), cmap='gray')
    axes[1].set_title("top serial")

    width = min(current_top_serial.shape[1], current_bottom_serial.shape[1])
    height = min(current_top_serial.shape[0], current_bottom_serial.shape[0])

    resized_bottom_serial = cv2.resize(current_bottom_serial, (width, height))
    resized_top_serial = cv2.resize(current_top_serial, (width, height))

    result = cv2.add(resized_bottom_serial, resized_top_serial)

    axes[2].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), cmap='gray')
    axes[2].set_title("After Add")

    reference_mean_saturation = np.mean(result)

    st.write(f"Real Serial_Intensity : 81.65454977429319")
    st.write(f"Current Serial_Intensity : {reference_mean_saturation}")

    if abs(reference_mean_saturation - 81.65454977429319) > threshold:
        fake = 1
        st.write("Serial check: Fake")
    else:
        st.write("Serial check: Real")

    plt.tight_layout()
    st.pyplot(fig)
    return fake

def check_fake(reference_image, current_image):
    if reference_image is None or current_image is None:
        st.error("Error loading images. Please check the file paths.")
        return "The Currency Is Fake --- Reason: Image Load Error"

    reference_left_lines, reference_bottom_serial, reference_top_serial = extract_features(reference_image)
    current_left_lines, current_bottom_serial, current_top_serial = extract_features(current_image)

    color = check_color(reference_image, current_image)
    bleedlines = check_bleedlines(reference_left_lines, current_left_lines)
    serial = check_serial(reference_bottom_serial[0], reference_top_serial[0], current_bottom_serial[0], current_top_serial[0])
    font = check_font(current_bottom_serial[0])

    if color:
        return "The Currency Is Fake --- Reason: Color Intensity"
    if bleedlines:
        return "The Currency Is Fake --- Reason: Bleedlines"
    if serial:
        return "The Currency Is Fake --- Reason: Serial no."
    if font:
        return "The Currency Is Fake --- Reason: Font Size"

    return "The Currency Is Real"

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO

import matplotlib as mpl
from cycler import cycler

palette = ['#1f2a29', '#4f291a', '#3c5f4e', '#75916b', '#dfde8c']
mpl.rcParams["axes.prop_cycle"] = cycler(color=palette)

def main():
    reference_image_path = 'data/real1.jpg'
    st.title("Fake Currency Checker")
    st.subheader("Upload an image")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg"])

    if uploaded_file is not None:
        try:
            reference_image = cv2.imread(reference_image_path)
            if reference_image is None:
                st.error("Error: Reference image not found or could not be read.")
                return

            file_bytes = uploaded_file.read()
            nparr = np.frombuffer(file_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is None:
                st.error("Error: Invalid or unsupported file format.")
                return

            output = check_fake(reference_image, image)
            st.write(output)
            st.image(image, caption="Uploaded Image", use_column_width=True)

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
