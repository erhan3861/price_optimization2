import numpy as np
import pandas as pd
from .apps import ApiConfig
from .data import getPriceFeatures, save_to_csv
from rest_framework.views import APIView
from rest_framework.response import Response
from django.shortcuts import render, HttpResponse
from django.contrib import messages
from price_optimization_API import settings

import django.template.loader as loader

# for download csv
import csv
from wsgiref.util import FileWrapper

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()


# our home page view
def home(request):
    return render(request, 'index_data.html', {})


def predict(request):
    return render(request, 'index.html', {})


# our result page view
def result(request):
    # ML_function()
    price_class = int(request.GET['price_class'])
    shipping = int(request.GET['shipping'])
    rating_point = int(request.GET['rating_point'])
    rating_number = int(request.GET['rating_number'])
    seller_point = int(request.GET['seller_point'])

    result = getPrediction(shipping, rating_point, rating_number, seller_point, price_class, request)

    return render(request, 'result.html', {'result': str(result[0])})


# custom method for generating predictions
def getPrediction(shipping, rating_point, rating_number, seller_point, price_class, request):
    import pickle
    model = pickle.load(open("price_prediction_model.sav", "rb"))
    scaled = pickle.load(open("scaler.sav", "rb"))
    global sc
    try:
        prediction = model.predict(sc.transform([[shipping, rating_point, rating_number, seller_point, price_class]]))

    except:
        print("hatalı")
        message_type = "warning"
        messages.warning(request, "Your dataset model needs to be trained first.")
        return render(request, 'index.html', {})

    return prediction


# get the dataset from the website
def get_data(request):
    # price classes are : 0-49 TL  50-99 TL ..... 450-499 TL
    price_class = 0

    num_class = request.GET['number_class']
    price_ranges = request.GET['price_ranges']

    if num_class == '':
        message_type = "warning"
        messages.warning(request, "Your class number is blank")
        return render(request, 'index_data.html', {})
    elif price_ranges == '':
        message_type = "warning"
        messages.warning(request, "Your pirce range  is blank")
        return render(request, 'index_data.html', {})
    else:
        message_type = "success"
        messages.success(request, "Your submit is succesful and you have e-commerce values")

    num_class = int(num_class)
    price_ranges = int(price_ranges)

    for i in range(num_class):
        print((i + 1), ".class")
        price_class += 1
        for j in range(1, 11, 1):
            # we are visiting all the desired pages
            print("     ", j, ".page")
            getPriceFeatures(request,
                             "https://www.n11.com/spor-giyim-ve-ayakkabi/spor-ayakkabi?q=spor+ayakkab%C4%B1&srt=SALES_VOLUME&minp=" + str(
                                 i * price_ranges) + "&maxp=" + str(
                                 i * price_ranges + price_ranges) + "&ref=auto&pg=" + str(j), price_class)

    # function for saving the data to csv file
    save_to_csv()
    return render(request, 'index_data.html', {})


import csv
from .data import get_dict


def getfile(request):
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="file.csv"'
    writer = csv.writer(response)
    data_dict = get_dict()
    print(len(data_dict))
    #print(len(data_dict['product_name']))

    for i in range(len(data_dict['product_name'])):
        writer.writerow([data_dict['product_name'][i], data_dict['new_price'][i], data_dict['old_price'][i],
                         data_dict['discount_ratio'][i], data_dict['shipping'][i], data_dict['rating_point'][i],
                         data_dict['rating_number'][i],
                         data_dict['seller_name'][i], data_dict['seller_point'][i], data_dict['price_class'][i]])
    return response


from sklearn.preprocessing import LabelEncoder
# Distribution plot on price
import seaborn as sns
from matplotlib import pyplot as plt


def ML_function(request):
    # import the data saved as a csv
    df = pd.read_csv("./api/price_dynamics.csv")
    # D:\price_optimization\price_dynamics2.csv
    print("df.heads", df.head())

    le = LabelEncoder()
    le_shipping = le.fit_transform(df['shipping'])
    le_seller_name = le.fit_transform(df['seller_name'])

    df['shipping'] = le_shipping
    df['seller_name'] = le_seller_name

    print(df.head())
    print(df.info())

    correlations = df.corr()

    point = []
    rating_number = []
    discount_ratio = []
    seller_point = []

    print(df.head())

    # veri kümesi
    # shipping,rating_point,rating_number,seller_point,price_class

    X = df.iloc[:, [4, 5, 6, 8, 9]].values
    y = df.iloc[:, 1].values

    # eğitim ve test kümelerinin bölünmesi
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Ölçekleme
    from sklearn.preprocessing import StandardScaler
    global sc
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)

    import numpy as np
    from sklearn.ensemble import RandomForestRegressor

    regressor = RandomForestRegressor(n_estimators=1000, random_state=42)
    regressor.fit(X_train, y_train)

    # saving model as a pickle
    import pickle
    pickle.dump(regressor, open("price_prediction_model.sav", "wb"))
    pickle.dump(sc, open("scaler.sav", "wb"))

    return render(request, 'index.html', {})


"""

class PricePrediction2(APIView):
    def post(self, request):
        data = request.data
        keys = []
        values = []
        for key in data:
            keys.append(key)
            values.append(data[key])
        
            
        X_test = pd.Series(values).to_numpy().reshape(1, -1)
        X_test = sc.transform(X_test) 
        
        randomForest_reg_model = ApiConfig.model
        y_pred = randomForest_reg_model.predict(X_test)
        y_pred = pd.Series(y_pred)
        response_dict = {"Predicted Price = ": y_pred[0]}
        return Response(response_dict, status=200)
"""
