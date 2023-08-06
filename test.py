import pytesseract
pytesseract.pytesseract.tesseract_cmd='C:/Program Files/Tesseract-OCR/tesseract'
import argparse
import cv2
import re

image=cv2.imread('bill.png',0)
text=(pytesseract.image_to_string(image)).lower()
match=re.findall(r'\d+[/.-]\d+[/.-]\d+', text)
st=" "
st=st.join(match)
print(st)
price=re.findall(r'[\$\£\€](\d+(?:\.\d{1,2})?)',text)
price=list(map(float,price))
print(max(price))
x=max(price)
