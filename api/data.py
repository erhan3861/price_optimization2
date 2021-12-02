from bs4 import BeautifulSoup
import requests
import time
import re
import math
import csv

import numpy as np
import pandas as pd

# this code blocks prevent our code from connection error to the website
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

from django.shortcuts import render, HttpResponse

url = "https://www.n11.com/spor-giyim-ve-ayakkabi/spor-ayakkabi?q=spor+ayakkab%C4%B1&srt=SALES_VOLUME&minp=1&maxp=50&ref=auto"

# main section
productList = []
new_priceList = []
old_priceList = []
ratioList = []
shippingList = []
ratingList = []
rating_textList = []
seller_nameList = []
seller_pointList = []
price_classList = []

data_dict = {}


# base function, we are getting features of the price classes
# this section is data-mining section

def reset_all_the_list():
    productList.clear()
    new_priceList.clear()
    old_priceList.clear()
    ratioList.clear()
    shippingList.clear()
    ratingList.clear()
    rating_textList.clear()
    seller_nameList.clear()
    seller_pointList.clear()
    price_classList.clear()


def getPriceFeatures(request, url, price_class):
    # for security prevent codes from the bots
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.67 Safari/537.36 Edg/87.0.664.55"
    }

    session = requests.Session()
    retry = Retry(connect=3, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)

    page = session.get(url, headers=headers)
    # print(page)
    htmlPage = BeautifulSoup(page.content, 'html.parser')
    data_all_color = htmlPage.find_all("li", attrs={"class": "column"})
    # print(data_all_color)

    for required_info in data_all_color:

        class_product_name = str(request.GET['class_product_name'])  # productName
        div_product_name = str(request.GET['div_product_name'])
        product_name = required_info.find(div_product_name, attrs={"class": class_product_name})  # "h3" 'productName'

        if product_name is None:
            product_name = required_info.find("h3", attrs={"class": "adGroupProduct"})

        product_name = product_name.text.strip()
        # print(product_name)
        productList.append(product_name)

        # old_price = request.GET['del']
        class_old_price = request.GET['class_old_price']
        div_old_price = request.GET['div_old_price']
        old_price = required_info.select(div_old_price)  # 'del'

        if old_price == []:
            old_priceList.append(0.00)
        else:
            for i in old_price:
                price = i.get_text()
                old_price = float(re.split(r'[\"]?([0-9\.]*)[\"]?', price)[1])
                old_priceList.append(old_price)
                # print(old_price)

        # new_price = request.GET['ins']
        div_new_price = request.GET["div_new_price"]
        class_new_price = request.GET["class_new_price"]
        # new_price = required_info.find(div_new_price,attrs={"class":class_new_price})
        new_price = required_info.find(div_new_price).text.strip()  # 'ins'

        new_price = float(re.split(r'[\"]?([0-9\.]*)[\"]?', new_price[:5])[1])
        new_priceList.append(float(new_price))

        # ratio = required_info.select('span[class="ratio"]') #ratio
        div_ratio = request.GET["div_ratio"]
        class_ratio = request.GET["class_ratio"]
        ratio = required_info.find(div_ratio, attrs={"class": class_ratio})

        if ratio == None:
            # print("0")
            ratioList.append(0)
        else:
            for i in ratio:
                # print(i.get_text())
                ratioList.append(int(i.get_text()))

        # shipping = required_info.select('span[class="textImg freeShipping"]')  #fat span,  textImg freeShipping
        div_shipping = request.GET["div_shipping"]
        class_shipping = request.GET["class_shipping"]
        shipping = required_info.find(div_shipping, attrs={"class": class_shipping})
        if shipping == None:
            # print("no shipping")
            shippingList.append("noShipping")
        else:
            shippingList.append("freeShipping")

        # rating_need = required_info.find('div',attrs={"class":"ratingCont"})  #div  ->  ratingCont
        div_rating = request.GET["div_rating"]
        class_rating = request.GET["class_rating"]
        rating_point = required_info.find(div_rating, attrs={"class": class_rating})
        tags = []
        if rating_point is None:
            # print("no rating")
            ratingList.append(0)
        else:
            point = rating_point.select('span')
            tags.extend((i.prettify() for i in point))
            print(tags)
            result_rating = tags[0]
            result_rating = result_rating[21:24]
            if result_rating != "100":
                result_rating = result_rating[:-1]
            ratingList.append(int(result_rating))

        div_rating_text = request.GET["div_rating_text"]
        class_rating_text = request.GET["class_rating_text"]

        if rating_point is not None:  # fat   span, ratingText
            rating_text = required_info.find(div_rating_text, attrs={"class": class_rating_text}).text.strip()

            if rating_text[2] == ",":
                st = ""
                for ch in rating_text:
                    if ch != ",":
                        st += ch
                rating_text = st
            # print(rating_text[1:-1])
            rating_textList.append(int(rating_text[1:-1]))
        else:
            rating_textList.append(0)

        div_seller_name = request.GET["div_seller_name"]
        class_seller_name = request.GET["class_seller_name"]
        seller_name = required_info.find(div_seller_name, attrs={"class": class_seller_name})  # fat   span, sallerName
        if seller_name is None:
            seller_name = required_info.find('span', attrs={"class": "adGroupSeller"})
        seller_name = seller_name.text.strip()
        seller_nameList.append(seller_name)

        div_seller_point = request.GET["div_seller_point"]
        class_seller_point = request.GET["class_seller_point"]
        seller_point = required_info.find(div_seller_point, attrs={"class": class_seller_point})  # fat   span, point
        if seller_point is not None:
            seller_point = seller_point.text.strip()
        else:
            seller_point = "%0"
        seller_pointList.append(seller_point[1:])
        # print(seller_point)

        price_classList.append(price_class)


def save_to_csv():
    data_headers = ['index', 'product_name', 'new_price', 'old_price', 'discount_ratio', 'shipping', 'rating_point',
                    'rating_number', 'seller_name', 'seller_point', 'price_class']
    global data_dict
    data_dict = {'product_name': productList, 'new_price': new_priceList, 'old_price': old_priceList,
                 'discount_ratio': ratioList, 'shipping': shippingList, 'rating_point': ratingList,
                 'rating_number': rating_textList, 'seller_name': seller_nameList,
                 'seller_point': seller_pointList, "price_class": price_classList}

    print("len of the data_dict after saving to csv", len(data_dict))

    with open('api/price_dynamics.csv', 'w', encoding='utf-8-sig', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            ['product_name', 'new_price', 'old_price', 'discount_ratio', 'shipping', 'rating_point', 'rating_number',
             'seller_name', 'seller_point', 'price_class'])
        writer.writerows(zip(*data_dict.values()))


def get_dict():
    print("len of the data_dict returned= ", len(data_dict))
    return data_dict


def get_product_list(list):
    global productList
    productList = list
    return productList



def find_words():
    global productList
    lower_product_list = productList
    others_list = []
    productListClear = []

    for i in range(len(lower_product_list)):
        lower_product_list [i] = lower_product_list[i].lower()
        
    for product in lower_product_list:
        if ("terli" in product) or ("terlik" in product) or ("boya" in product) or ("ipli" in product):
            others_list.append(product)
        else:
            productListClear.append(product)
    
    #print("others = ",len(others_list))
    #print(others_list)
    #print("---------------------------------------------------------------")
    #print("White list of the product = ",len(productListClear))
    #print(productListClear)

    import nltk 
    from TurkishStemmer import TurkishStemmer
    
    stemmer = TurkishStemmer()
    wordfreq = {}

    for sentence in others_list:
        tokens = nltk.word_tokenize(sentence)
        for token in tokens:
            token = stemmer.stem(token)
            if token not in wordfreq.keys():
                wordfreq[token] = 1
            else:
                wordfreq[token] += 1

    #print(wordfreq)

    wordfreqTargetProduct = {}
    for sentence in productListClear:
        tokens = nltk.word_tokenize(sentence)
        for token in tokens:
            token = stemmer.stem(token)
            if token not in wordfreqTargetProduct.keys():
                wordfreqTargetProduct[token] = 1
            else:
                wordfreqTargetProduct[token] += 1

    #print(wordfreqTargetProduct)

    import heapq
    most_freq = heapq.nlargest(200, wordfreq, key=wordfreq.get)
    most_freq_target = heapq.nlargest(200, wordfreqTargetProduct, key=wordfreqTargetProduct.get)


    most_freq_target_counts = []
    for item in most_freq_target:
        most_freq_target_counts.append(wordfreqTargetProduct[item])

    sentence_vectors = []
    for sentence in productListClear:
        sentence_tokens = nltk.word_tokenize(sentence)
        sent_vec = []
        for token in most_freq_target:
            if token in sentence_tokens:
                sent_vec.append(1)
            else:
                sent_vec.append(0)
        sentence_vectors.append(sent_vec)

    sentence_vectors = np.asarray(sentence_vectors)
    type(sentence_vectors)
    #print()

    df_most_targetwords = pd.DataFrame(most_freq_target,columns=["words"])
    df_most_targetwords["counts"] = most_freq_target_counts
    #print("most used words are:")
    #print(df_most_targetwords)

    df_most_targetwords = df_most_targetwords.to_dict('split')
    #print(df_most_targetwords)


    table_dict_1 = {}
    table_dict_2 = {}
    counter = 0
    for item in  df_most_targetwords["data"]:     
        if counter<100:       
            table_dict_1[item[0]] = item[1]
        else:
            table_dict_2[item[0]] = item[1]
        counter += 1

    #print(table_dict)


    return table_dict_1, table_dict_2


