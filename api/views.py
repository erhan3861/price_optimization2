import numpy as np
import pandas as pd

import price_optimization_API
from .apps import ApiConfig
from .data import url, find_words, getPriceFeatures, save_to_csv, reset_all_the_list, get_product_list,find_words
from rest_framework.views import APIView
from rest_framework.response import Response
from django.shortcuts import render, HttpResponse
from django.http import JsonResponse
from django.contrib import messages
from price_optimization_API import settings

import django.template.loader as loader

# for download csv
import csv
from wsgiref.util import FileWrapper
from django.views.decorators.csrf import csrf_protect, csrf_exempt

from sklearn.preprocessing import StandardScaler
from urllib.parse import unquote

sc = StandardScaler()

#for nltk issues


#global vars
price_ranges = "50"
num_class = "10"
product_url = ""
store_name = "n11"
price_class = 0
store_list = []


# our home page view
def home(request):
    context = {
            "info_for_radio_btn" : "Choose radio button 1 and then GET VALUES button."
            }
    return render(request, 'index_data.html', context)


def predict(request):
    return render(request, 'index.html', {})


def help(request):
    return render(request, 'help.html', {})


#set the store 
def set_store(request):
     global product_url,store_name
     #store_name = request.GET.get('store_name', "N11")
     #product_url = request.GET.get('product_url', "N11")
     print(store_name)
     return render(request, 'index_data.html', {})


# our result page view

def result(request):
    # ML_function()
    
    price_class = int(request.POST.get('price_class'))
    shipping = int(request.POST.get('shipping',"0"))
    rating_point = int(request.POST.get('rating_point',"50"))
    rating_number = int(request.POST.get('rating_number',"10"))
    seller_point = int(request.POST.get('seller_point',"50"))

    print(str(price_class),",",str(shipping),"",str(rating_point),",",str(rating_number),",",str(seller_point))

    result = getPrediction(shipping, rating_point, rating_number, seller_point, price_class, request)
    
    most_words_1, most_words_2, man_woman_dict, brand_dict, sport_dict, color_dict = find_words()

    #get the product from url
    if len(store_list) > 0:
        product = store_list[0]
        product.index('=')
        product.index('&')
        product = product[product.index('=')+1:product.index('&')]
        product = unquote(product)
    else:
        product = "spor ayakkabı"
    print("product = ",product)

    context = {
        'result': result[0],
        'most_words_1':most_words_1, 
        'most_words_2':most_words_2,
        'man_woman_dict' : man_woman_dict, 
        'brand_dict': brand_dict, 
        'sport_dict' : sport_dict, 
        'color_dict' : color_dict,
        'product' : product,
    }
    return render(request, 'result.html', context)
    # return render(request, 'result.html', {'result': result[0],'most_words_1':most_words_1, 'most_words_2':most_words_2})


# custom method for generating predictions
def getPrediction(shipping, rating_point, rating_number, seller_point, price_class, request):
    import pickle
    model = pickle.load(open("./price_prediction_model.sav", "rb"))
    #scaled = pickle.load(open("./scaler.sav", "rb"))
    global sc
    try:
        prediction = model.predict(sc.transform([[shipping, rating_point, rating_number, seller_point, price_class]]))

    except:
        print("hatalı")
        message_type = "warning"
        messages.warning(request, "Your dataset model needs to be trained first.")
        return render(request, 'index.html', {})

    return prediction



def continue_get_data(request, class_value, store_name, product_url, price_class):
    global price_ranges 


    if class_value == "34":
        num_class = 3
    elif class_value == "56":
        num_class = 5
    elif class_value == "78":
        num_class = 7
    elif class_value == "910":
        num_class = 9

    for i in range(num_class-1, num_class+1, 1):
        print((i), ".class 2nd side", store_name, product_url)
        price_class += 1
        for j in range(1, 11, 1):
            # we are visiting all the desired pages
            print("     ", j, ".page")
            if(store_name == "N11"):
                getPriceFeatures(request, store_name, 
                                product_url +"&srt=SALES_VOLUME&minp=" + str(
                                    i * price_ranges) + "&maxp=" + str(
                                    i * price_ranges + price_ranges) + "&ref=auto&pg=" + str(j), price_class)
            elif(store_name == "Gittigidiyor"):
                # addition = &fmax=100&fmin=50&sf=2
                getPriceFeatures(request, store_name,
                                product_url +"&fmax="+str(
                                    i * price_ranges + price_ranges)
                                    + "&fmin=" + str(
                                    i * price_ranges) + "&sf=" + str(j), price_class)
            
            elif(store_name == "Trendyol"):
                # addition = &fmax=100&fmin=50&sf=2
                getPriceFeatures(request, store_name,
                                product_url +"&prc="+str(
                                    i * price_ranges)
                                    + "-" + str(
                                    i * price_ranges + price_ranges) + "&pi=" + str(j), price_class)
    store_list[2] = price_class
    save_to_csv()
    return price_class



# get the dataset from the website
def get_data(request):
    # price classes are : 0-49 TL  50-99 TL ..... 450-499 TL
    
   
    #clear all the lists for getting new csv file
    class_value = request.POST.get('sub_class_name')
    print("class_value = ", class_value,  "store_list = ",store_list)

    if class_value == "12":
        reset_all_the_list(store_list)
        price_class = 0
    elif class_value != "12":
        product_url = store_list[0]
        store_name = store_list[1]
        price_class = store_list[2]
        print("price_class=", price_class,"product_url=",product_url,"-","store_name",store_name, "last_class",store_list[3] )
        last_price_class = continue_get_data(request, class_value, store_name, product_url, price_class)
        if last_price_class != store_list[3]:  
            context = {
            "info_for_radio_btn" : "Class  "+ str(last_price_class-1) +" - "+str(last_price_class)+" finished.Click  "
            +str(str(last_price_class//2+1))+ " radio button then GET VALUES button."
            }
        else:
            print(store_list[3])
            context = {
            "info_for_radio_btn" : "Class  "+ str(last_price_class-1) +" - "+str(last_price_class)
            +"finished \n Click  RED TRAIN BUTTON on the up for training"
            }
        return render(request, 'index_data.html', context)
  
    #store_name = request.GET['store_name']
    #product_url = request.GET['product_url']
    store_name = request.POST.get('store_name')
    product_url = request.POST.get('product_url')
    print(store_name,"pr-name= ",product_url)
    
    store_list.append(product_url)
    store_list.append(store_name)
    
    global num_class,price_ranges
    total_num_class = request.POST.get('number_class')
    price_ranges = request.POST.get('price_ranges')

    print("num_class=",num_class)
    print("price_ranges=",price_ranges)


    if total_num_class == '':
        message_type = "warning"
        messages.warning(request, "Your class number is blank")
        return render(request, 'index_data.html', {})
    elif price_ranges == '':
        message_type = "warning"
        messages.warning(request, "Your price range  is blank")
        return render(request, 'index_data.html', {})
    else:
        message_type = "success"
        messages.success(request, "Your submit is succesful and you have e-commerce values")

    if class_value == "12":
        num_class = 1


    num_class = int(num_class)
    price_ranges = int(price_ranges)

    for i in range(num_class-1, num_class+1, 1):
        print((i + 1), ".class")
        price_class += 1
        for j in range(1, 11, 1):
            # we are visiting all the desired pages
            print("     ", j, ".page")
            if(store_name == "N11"):
                getPriceFeatures(request, store_name, 
                                product_url +"&srt=SALES_VOLUME&minp=" + str(
                                    i * price_ranges) + "&maxp=" + str(
                                    i * price_ranges + price_ranges) + "&ref=auto&pg=" + str(j), price_class)
            elif(store_name == "Gittigidiyor"):
                # addition = &fmax=100&fmin=50&sf=2
                getPriceFeatures(request, store_name,
                                product_url +"&fmax="+str(
                                    i * price_ranges + price_ranges)
                                    + "&fmin=" + str(
                                    i * price_ranges) + "&sf=" + str(j), price_class)
            
            elif(store_name == "Trendyol"):
                # addition = &fmax=100&fmin=50&sf=2
                getPriceFeatures(request, store_name,
                                product_url +"&prc="+str(
                                    i * price_ranges)
                                    + "-" + str(
                                    i * price_ranges + price_ranges) + "&pi=" + str(j), price_class)
    store_list.append(price_class)
    store_list.append(int(total_num_class))
    

    # function for saving the data to csv file
    save_to_csv()
    contex = {"info_for_radio_btn":" Class 1-2 ready. \n Click 2nd radio button"}
    return render(request, 'index_data.html', contex)


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
    print("here...",request.POST)

    global price_ranges
    if request.POST['valid_price_class'] is not ['']:
        price_ranges = request.POST['valid_price_class']
        store_text = request.POST['valid_store_name']
    else :
        df = pd.read_csv("./api/price_dynamics.csv")

    print("type0=",type(price_ranges),"   price_ranges = ",price_ranges,"store_text",store_text )

    
    if str(price_ranges) == '25' and store_text=='N11':
        df = pd.read_csv("./api/price_dynamics_25.csv")
    elif str(price_ranges) == '25' and store_text=='Trendyol':
        df = pd.read_csv("./api/price_dynamics_25_T.csv")
    elif str(price_ranges) == '50'and store_text=='N11':
        df = pd.read_csv("./api/price_dynamics_50.csv")
    elif str(price_ranges) == '50' and store_text=='Trendyol':
        df = pd.read_csv("./api/price_dynamics_50_T.csv")   
    elif str(price_ranges) == '75'and store_text=='N11':
        df = pd.read_csv("./api/price_dynamics_75.csv")
    elif str(price_ranges) == '75' and store_text=='Trendyol':
        df = pd.read_csv("./api/price_dynamics_75_T.csv")  
    elif str(price_ranges) == '100'and store_text=='N11':
        df = pd.read_csv("./api/price_dynamics_100.csv")
    elif str(price_ranges) == '100' and store_text=='Trendyol':
        df = pd.read_csv("./api/price_dynamics_100_T.csv")  
    else:
        print("df get values is entered")
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
    X_test = sc.fit_transform(X_test)

    import numpy as np
    from sklearn.ensemble import RandomForestRegressor

    regressor = RandomForestRegressor(n_estimators=1000, random_state=42)
    regressor.fit(X_train, y_train)

    #actual and prediction visualition
    import seaborn as sns
    plt.figure(figsize=(6, 6))

    y_pred = regressor.predict(X_test)
    #for comparison actual and predicted values
    df_actual_predicted = pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
    print(df_actual_predicted )
    
    ax = sns.distplot(y, hist=False, color="r", label="Actual Value")
    sns.distplot(y_pred, hist=False, color="b", label="Fitted Values" , ax=ax)


    plt.title('Actual vs Fitted Values for Price')
    plt.xlabel('price_range')
    plt.ylabel('density')
    plt.legend(loc='upper right',bbox_to_anchor=(1,0.95))
    plt.savefig("./api/static/images/actual_fitted.png")
    #plt.show()
    #plt.close()

    # saving model as a pickle
    import pickle
    pickle.dump(regressor, open("price_prediction_model.sav", "wb"))
    pickle.dump(sc, open("scaler.sav", "wb"))
    
    product_name_list = df.iloc[:, 0].values.tolist()
    print(type(product_name_list))
    get_product_list(product_name_list)
    

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
