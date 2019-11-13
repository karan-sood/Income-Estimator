from django.shortcuts import render
from . import predict_income

def home(request):
    return render(request, 'index.html')

def result(request):
    age = int(request.GET["age"])
    gender = request.GET["gender"]
    postcode = request.GET["postcode"]
    predict = predict_income.getIncome(age, gender, postcode)
    return render(request, 'result.html', {'prediction': predict})
