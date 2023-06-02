from django.shortcuts import render
import numpy as np
import pickle
import os



# Create your views here.
def index(request):
    return render(request, "home.html")



def prediction(request):
    # test = os.getcwd()
    # print("djeldi", test)
    model = pickle.load(open("static\models\iris_model_svc.pkl", "rb"))
    sl, sw, pl, pw = request.POST["sl"], request.POST["sw"], request.POST["pl"], request.POST["pw"]
    # print("sl: ", sl, "sw: ", sw, "pl: ", pl, "pw: ", pw)
    f_data = np.array([[sl, sw, pl, pw]], dtype=float)
    y_pred = model.predict(f_data)
    
    return render(request, "after.html", {"data": y_pred})