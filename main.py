import os
from flask import Flask, request, redirect, render_template, flash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing  import image

import numpy as np
import random

classes = ["青眼の白龍(ブルーアイズ)","ブラックマジシャン","真紅眼の黒竜(レッドアイズ)","暗黒騎士ガイア","究極完全態・グレート・モス"]
image_size = 50

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

model = load_model('./my_new_model.h5')#学習済みモデルをロード

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('ファイルがありません')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('ファイルがありません')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename))
            filepath = os.path.join(UPLOAD_FOLDER, filename)

            #受け取った画像を読み込み、np形式に変換
            img = image.load_img(filepath, grayscale=False, target_size=(image_size,image_size))
            print(img)
            img = image.img_to_array(img)
            print(img)
            data = np.array([img])
            print(data)
            #変換したデータをモデルに渡して予測する
            result = model.predict(data)[0]
            predicted = result.argmax()
            num = random.randint(0,60)
            if num < 2:
                result = "大吉"
            elif 2 <= num < 10:
                result = "中吉"
            elif 10 <= num < 20:
                result = "小吉"
            elif 20 <= num < 40:
                result = "吉"
            elif 40 <= num < 50:
                result = "末吉"
            elif 50 <= num < 55:
                result = "凶"
            elif 55 <= num < 58:
                result = "中凶"
            else:
                result = "大凶"
            pred_answer = "これは " + classes[predicted] + " です" + "\n"+ "あなたの今日の運勢は" + result + "です"

            return render_template("index.html",answer=pred_answer)

    return render_template("index.html",answer="")


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    app.run(host ='0.0.0.0',port = port)

