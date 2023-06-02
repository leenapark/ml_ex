# pip install flask
# flask server 실행 : 
from flask import Flask, render_template, request
import pickle
import numpy as np



# Flask(__name__)를 호출하여 Flask app 초기화
app = Flask(__name__)

# pickle ML 모델 로드
model = pickle.load(open("models\iris_model_svc.pkl", "rb"))

# http://127.0.0.1:5000
# def index() 정의
@app.route('/')
def index():
    return render_template("home.html")


# http://127.0.0.1:5000/predict
# url path : predict
@app.route('/predict', methods=["POST"])
def prediction():
    # 꽃받침 길이값
    print("예외처리", request.form["sl"])
    sl = request.form["sl"]
    # 꽃받침 너비값
    sw = request.form["sw"]
    # 꽃잎 길이값
    pl = request.form["pl"]
    # 꽃잎 너비값
    pw = request.form["pw"]
    # sl, sw, pl, pw = float(sl), float(sw), float(pl), float(pw)
    """
    1. 클라이언트에서 데이터를 request 받음
    2. 각각의 데이터(form input에 입력한 값)을 변수에 저장
    3. numpy array 로 데이터 타입 같게 만들기(train_X와 동일한 type)
    4. ML model 을 로딩해서, predict() 수행
    5. 예측 수행 결과를 변수에 저장
    6. 예측 값을 클라이언트에게 response 해줌
      a. 예측 데이터를 html 화면으로 rendering -> response
    """
    f_data = np.array([[sl, sw, pl, pw]], dtype=float)
    y_pred = model.predict(f_data)
    # print(y_pred, type(y_pred))
    # print("sl: ", sl, "sw: ", sw, "pl: ", pl, "pw: ", pw)
    # print("type: ", type(sl), "type: ", type(sw), "type: ", type(pl), "type: ", type(pw))

    # 문법
    # return render_template("html template", data=data 변수명)
    return render_template("after.html", data=y_pred)

if __name__ == '__main__':
    app.run(host="127.0.0.1", port="5000", debug=True)