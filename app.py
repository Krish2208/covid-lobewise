from flask import Flask, render_template, request
from PIL import Image
import numpy as np
from model.segmentation import predict, model, ggo_detect
import cv2
import io
import base64

app = Flask(__name__)


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

# @app.route('/about', methods=['GET'])
# def about():
#    return render_template('about.html')


def convertImgtoBase64(img):
    image = (img * 255).astype(np.uint8)
    image = Image.fromarray(image)
    data = io.BytesIO()
    image.save(data, "jpeg")
    encoded_img = base64.b64encode(data.getvalue())

    return encoded_img


@app.route('/upload', methods=['POST'])
def im_render():
    image = request.files['chest']
    image = np.array(Image.open(image.stream).resize(
        (512, 512)).convert('RGB'))
    mask = cv2.imread('./static/images/mask.png')
    output = predict(image)
    new = ggo_detect(image, output)
    return render_template('output.html',
                           img_1_br=convertImgtoBase64(
                               new[0][0]).decode('utf-8'),
                           img_1_op=convertImgtoBase64(
                               new[0][1]).decode('utf-8'),
                           img_2_br=convertImgtoBase64(
                               new[1][0]).decode('utf-8'),
                           img_2_op=convertImgtoBase64(
                               new[1][1]).decode('utf-8'),
                           img_3_br=convertImgtoBase64(
                               new[2][0]).decode('utf-8'),
                           img_3_op=convertImgtoBase64(
                               new[2][1]).decode('utf-8'),
                           img_4_br=convertImgtoBase64(
                               new[3][0]).decode('utf-8'),
                           img_4_op=convertImgtoBase64(
                               new[3][1]).decode('utf-8'),
                           img_5_br=convertImgtoBase64(
                               new[4][0]).decode('utf-8'),
                           img_5_op=convertImgtoBase64(
                               new[4][1]).decode('utf-8'),
                           img_6_br=convertImgtoBase64(
                               new[5][0]).decode('utf-8'),
                           img_6_op=convertImgtoBase64(
                               new[5][1]).decode('utf-8'),
                           )


if __name__ == '__main__':
    app.run(port=5000, debug=True)
