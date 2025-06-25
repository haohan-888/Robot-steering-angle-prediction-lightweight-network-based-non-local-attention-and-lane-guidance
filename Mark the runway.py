import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

input_folder = r"D:\mypycharm\pythonProject5\log11"
output_folder = r"D:\mypycharm\pythonProject5\log12"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        original_path = os.path.join(output_folder, f"{filename}_original.jpg")
        cv2.imwrite(original_path, img)

        gray_path = os.path.join(output_folder, f"{filename}_gray.jpg")
        cv2.imwrite(gray_path, img)

        edges = cv2.Canny(img, threshold1=100, threshold2=200)

        edges_path = os.path.join(output_folder, f"{filename}_edges.jpg")
        cv2.imwrite(edges_path, edges)

        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)

        if lines is not None:
            all_lines_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(all_lines_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            all_lines_path = os.path.join(output_folder, f"{filename}_all_lines.jpg")
            cv2.imwrite(all_lines_path, all_lines_img)

        if lines is not None:
            filtered_lines_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if y1 > 47 and y2 > 30:  # 120 // 2 = 60
                    cv2.line(filtered_lines_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            filtered_lines_path = os.path.join(output_folder, f"{filename}_filtered_lines.jpg")
            cv2.imwrite(filtered_lines_path, filtered_lines_img)

        plt.figure(figsize=(20, 5))

        plt.subplot(1, 5, 1)
        plt.imshow(img, cmap='gray')
        plt.title("Original Image")
        plt.axis('off')

        plt.subplot(1, 5, 2)
        plt.imshow(img, cmap='gray')
        plt.title("Grayscale Image")
        plt.axis('off')

        plt.subplot(1, 5, 3)
        plt.imshow(edges, cmap='gray')
        plt.title("Canny Edges")
        plt.axis('off')

        if lines is not None:
            plt.subplot(1, 5, 4)
            plt.imshow(cv2.cvtColor(all_lines_img, cv2.COLOR_BGR2RGB))
            plt.title("All Detected Lines")
            plt.axis('off')

        if lines is not None:
            plt.subplot(1, 5, 5)
            plt.imshow(cv2.cvtColor(filtered_lines_img, cv2.COLOR_BGR2RGB))
            plt.title("Filtered Lines")
            plt.axis('off')

        plt.tight_layout()
        plt.show()

print("！")