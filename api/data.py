from bs4 import BeautifulSoup, dammit
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
url_g = "https://www.gittigidiyor.com/ayakkabi/spor-ayakkabi?k=spor+ayakkab%C4%B1&qm=1&fmax=50&sf=2"
#https://www.gittigidiyor.com/ayakkabi/spor-ayakkabi?k=spor%20ayakkab%C4%B1&qm=1
# terlik -> https://www.gittigidiyor.com/ayakkabi/terlik?k=terlik&qm=1
# addition = &fmax=100&fmin=50&sf=2


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

tag_list = []

class_product_name,div_product_name,class_old_price,div_old_price, div_new_price,class_new_price = "","","","","",""
div_ratio, class_ratio, div_shipping, class_shipping, div_rating, class_rating = "","","","","",""
div_rating_text, class_rating_text, div_seller_name, class_seller_name, div_seller_point, class_seller_point = "","","","","",""

data_dict = {}
longer_tags = "0"
#get the right url
def get_store_url(url):
    url = url
    print(url)

# base function, we are getting features of the price classes
# this section is data-mining section

def reset_all_the_list(store_list):
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
    tag_list.clear()
    store_list.clear()
    print("reset_all_the_list store_list=",store_list)

def getPriceFeatures(request, store_name, url, price_class):
    # for security prevent codes from the bots
    #return
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.67 Safari/537.36 Edg/87.0.664.55"
    }

    print(store_name, url, "class = ",price_class)

    session = requests.Session()
    retry = Retry(connect=3, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)

    page = session.get(url, headers=headers)
    #print(store_name)
    htmlPage = BeautifulSoup(page.content, 'html.parser')

    if store_name == "N11":
        data_all_color = htmlPage.find_all("li", attrs={"class": "column"})
        longer_tags = "0"
        #print(data_all_color,"data_all_color1")
        if data_all_color == []:
            data_all_color = htmlPage.find_all("li", attrs={"class": "itemize-piece"})
            longer_tags = "1"
            #print(data_all_color,"data_all_color")

    elif store_name == "Gittigidiyor":
        data_all_color = htmlPage.find_all("div", attrs={"class": "pmyvb0-0 jCCkZh"})
    elif store_name == "Trendyol":
        longer_tags = "99"
        data_all_color = htmlPage.find_all("div", attrs={"class": "p-card-wrppr"})

    #print(data_all_color)
    global class_product_name,div_product_name,class_old_price,div_old_price, div_new_price, class_new_price
    global div_ratio, class_ratio, div_shipping, class_shipping, div_rating, class_rating, div_rating_text, class_rating_text, div_seller_name, class_seller_name, div_seller_point, class_seller_point
    for required_info in data_all_color:
        #print("-------")
        if price_class == 1:
            div_product_name =  request.POST.get('div_product_name')
            tag_list.append(div_product_name)
            class_product_name =  request.POST.get('class_product_name')
            tag_list.append(class_product_name)
            div_old_price = request.POST.get('div_old_price')
            tag_list.append(div_old_price)
            # old_price = request.GET['del']
            class_old_price = request.POST.get('class_old_price')
            tag_list.append(class_old_price)
            # new_price = request.GET['ins']
            div_new_price = request.POST.get("div_new_price")
            tag_list.append(div_new_price)
            class_new_price = request.POST.get("class_new_price")
            tag_list.append(class_new_price)
            #discount ratio ratio = required_info.select('span[class="ratio"]') #ratio
            div_ratio = request.POST.get("div_ratio")
            tag_list.append(div_ratio)
            class_ratio = request.POST.get("class_ratio")
            tag_list.append(class_ratio)
            #shipping
            div_shipping = request.POST.get("div_shipping")
            tag_list.append(div_shipping)
            class_shipping = request.POST.get("class_shipping")
            tag_list.append(class_shipping)
            #rating point
            div_rating = request.POST.get("div_rating")
            tag_list.append(div_rating)
            class_rating = request.POST.get("class_rating")
            tag_list.append(class_rating)
            # rating count index = 12
            div_rating_text = request.POST.get("div_rating_text")
            tag_list.append(div_rating_text)
            class_rating_text = request.POST.get("class_rating_text")
            tag_list.append(class_rating_text)
            #seller name 
            div_seller_name = request.POST.get("div_seller_name")
            tag_list.append(div_seller_name)
            class_seller_name = request.POST.get("class_seller_name")
            tag_list.append(class_seller_name)
            #seller point
            div_seller_point = request.POST.get("div_seller_point")
            tag_list.append(div_seller_point)
            class_seller_point = request.POST.get("class_seller_point")
            tag_list.append(class_seller_point)
        
        
        if longer_tags == "1":
            product_name = required_info.find("h3", attrs={"class": "itemize-title"})  # "n11 longer tags" 'productName
        else:    
            # product_name = required_info.find(div_product_name, attrs={"class": class_product_name})  # "h3" 'productName'
            product_name = required_info.find(tag_list[0], attrs={"class": tag_list[1]})  # "h3" 'productName'
        
        #print(price_class,  "product_name=",product_name)
        if product_name is None:
            product_name = required_info.find("h3", attrs={"class": "adGroupProduct"})
            continue
            #product_name = required_info.select("h3")  # 'del'
 
        product_name = product_name.text.strip()
        #print(product_name)
        productList.append(product_name)
        

        
        
        if store_name == "N11":
            if longer_tags == "0":
                old_price = required_info.find(tag_list[2], attrs={"class": tag_list[3]})
                # old_price = required_info.find(div_old_price, attrs={"class": class_old_price})
            else:
                old_price = required_info.find("span", attrs={"class": "itemize-price-old"})
        elif store_name == "Trendyol":
            old_price = required_info.find(tag_list[2], attrs={"class": tag_list[3]})#div
            if old_price is None:
                old_price = 0.0
            else:
                old_price = old_price.text.strip()
        #print(old_price)
        old_price_temp = ""

        if old_price is None:
            #print("boş old_price liste entered")
            old_price = 0.0
            old_priceList.append(0.00)
        else:
            if store_name == "N11":
                if longer_tags == "0": 
                    for i in old_price:
                        price = i.get_text()
                        old_price = float(re.split(r'[\"]?([0-9\.]*)[\"]?', price)[1])
                else :
                    old_price = old_price.text.strip()
                    old_price = old_price.replace(".","")
                    old_price = old_price.replace(" TL","").replace(",",".")   #44,90 TL
                    old_price = old_price.replace("TL","")
                    old_price = float(old_price)

            elif store_name == "Trendyol":
                for ch in str(old_price):
                    if ch.isdigit():
                        old_price_temp += ch
                    elif ch == ",":
                        old_price_temp += "."
                    old_price = float(old_price_temp)
                
            #print("dolu  old_price  liste entered",old_price)
            old_priceList.append(old_price)
          

        
        # new_price = required_info.find(div_new_price,attrs={"class":class_new_price})
        if store_name == "N11":
            if longer_tags == "0": 
                # new_price = required_info.find(div_new_price, attrs={"class": class_new_price})
                new_price = required_info.find(tag_list[4], attrs={"class": tag_list[5]})
            else:
                new_price = required_info.find("p", attrs={"class": "itemize-price-accurate"})
            
            #print(new_price)
            if new_price is not None:
                new_price = new_price.text.strip()
                new_price = new_price.replace(".","")
                new_price = new_price.replace(",",".")
                new_price = new_price.replace(" ","").replace("\nTL","")
                new_price = new_price.replace("TL","")
                new_price = float(new_price)
                #print(new_price)
                #print("true new_price list entered",new_price)
            else:
                new_price = old_price
                #print("false new_price list entered",new_price)
            
        elif store_name == "Trendyol":
            
            new_price_temp = ""
            new_price = required_info.find(tag_list[4], attrs={"class": tag_list[5]})
            if new_price is not None:
                new_price = new_price.text.strip()
                for ch in str(new_price):
                    if ch.isdigit():
                        new_price_temp += ch
                    elif ch == ",":
                        new_price_temp += "."
                new_price = float(new_price_temp)
                #print("true new_price list entered",new_price)
            else:
                #print("false new_price list entered",new_price)
                new_price = old_price

        new_priceList.append(new_price)
        #print("old price = ",old_price,"new price = ",str(new_price))
        
        
        
        ratio = required_info.find(tag_list[6], attrs={"class": tag_list[7]})
    
        if ratio == None:
            # print("0")
            if store_name == "N11" and longer_tags == "0":
                ratioList.append(0)
            elif store_name == "Trendyol" or (store_name == "N11" and longer_tags == "1"):
                if old_price == 0.0:
                    ratioList.append(0)
                else:
                    ratioList.append(int(((old_price - new_price)/old_price)*100))  
        else:
            if store_name == "N11":
                if longer_tags == "0":
                    for i in ratio:
                        # print(i.get_text())
                        ratioList.append(int(i.get_text()))

        
        #print(ratioList)
        #shipping = required_info.select('span[class="textImg freeShipping"]')  #fat span,  textImg freeShipping
        
        shipping = required_info.find(tag_list[8], attrs={"class": tag_list[9]})
        if longer_tags == "1":
                    shipping = required_info.find("span", attrs={"class": "itemize-cargo-badge itemize-cargo-free"})

        if shipping == None:
            # print("no shipping")
            shippingList.append("noShipping")
        else:
            shippingList.append("freeShipping")

        #print(shippingList)
        
        #start of the rating_points
        # rating_need = required_info.find('div',attrs={"class":"ratingCont"})  #div  ->  ratingCont
        
        
        if store_name == "N11":
            if longer_tags == "0":
                rating_point = required_info.find(tag_list[10], attrs={"class": tag_list[11]}) #for the sport shoes
            else:
                rating_point = required_info.find("i", attrs={"class": "rating_stars--black"}) #for general products
            tags = []
            #print(rating_point)
            if rating_point is None:
                # print("no rating")
                ratingList.append(0)
            else:
                if longer_tags == "0":
                    point = rating_point.select('span')
                    tags.extend((i.prettify() for i in point))
                    #print(tags)
                    result_rating = tags[0]
                    result_rating = result_rating[21:24]
                    if result_rating != "100":
                        result_rating = result_rating[:-1]
                    ratingList.append(int(result_rating))

                else:
                    # for item in rating_point:
                    point = rating_point["style"]
                    point = point.replace("width: ","")   #width: 90%
                    point = point.replace("%","")
                    ratingList.append(int(point))
                         

        elif store_name == "Trendyol":
            rating_point = required_info.find(tag_list[10], attrs={"class": tag_list[11]})
            if rating_point is None:
                # print("no rating")
                ratingList.append(0)
            else:
                average_point = 0
                for item in rating_point:
                    #print("item = ", item)
                    # print("find = ",item.find("div", attrs={"class": "star"}))
                    # if item.find("div", attrs={"class": "star"}) is None:
                    #     print("find = ",item.find("div", attrs={"class": "star"}))
                    #     continue 
                    point = 0
                    # point = item["style"].split(";")
                    style_class = item.find("div", attrs={"class": "full"}) 
                    if style_class is None:
                        break
                    point = style_class["style"].split(";")
                    #print("point",point)
                    point = point[0]
                    point = point.replace("width:","")
                    point = point.replace("%","")
                    average_point += int(point)
                ratingList.append(average_point//5)
                #print(average_point//5)
        #print(ratingList)  
 

        if rating_point is not None:  # fat   span, ratingText
            rating_text = required_info.find(tag_list[12], attrs={"class": tag_list[13]})  
            if store_name == "N11":
                if longer_tags == "0":
                    rating_text = rating_text.text.strip()
                    if rating_text[2] == ",":
                        st = ""
                        for ch in rating_text:
                            if ch != ",":
                                st += ch
                        rating_text = st
                    # print(rating_text[1:-1])
                    rating_textList.append(int(rating_text[1:-1]))
                if longer_tags == "1":
                    rating_point = required_info.find("p", attrs={"class": "itemize-star-count"}) 
                    rating_point = rating_point.text.strip()
                    rating_point = rating_point.replace("(","").replace(")","").replace(",","")
                    rating_textList.append(int(rating_point))  

            elif store_name == "Trendyol":
                rating_text = rating_text.text.strip()
                rating_text = rating_text.replace("(","")
                rating_text = rating_text.replace(")","")
                rating_textList.append(int(rating_text))
        else:
            rating_textList.append(0)        
        #end of the rating count
        #print(rating_textList)
       
        #seller name
        seller_name = required_info.find(tag_list[14], attrs={"class": tag_list[15]})  # fat   span, sallerName
        #print(seller_name)
        if seller_name is not None:
            seller_name = seller_name.text.strip()

        if seller_name is None:
            if store_name == "N11":
                if longer_tags == "0":
                    seller_name = required_info.find('span', attrs={"class": "adGroupSeller"})
                    if seller_name is not None:
                        seller_name = seller_name.text.strip()
                    else:
                        seller_name = "No record"
                elif longer_tags == "1":
                    seller_name = product_name.split()
                    seller_name = seller_name[0]
            elif store_name == "Trendyol":
                seller_name = "None"
        #elif seller_name is not None and store_name == "Trendyol":
            #seller_name = seller_name.text.strip()

        seller_nameList.append(seller_name)
        #end of the seller name
        #print(seller_nameList)
        
        #seller point
        seller_point = required_info.find(tag_list[16], attrs={"class": tag_list[17]})  # fat   span, point
        if store_name == "N11": 
            if seller_point is not None:
                if longer_tags == "0":
                    seller_point = seller_point.text.strip()
            else:
                seller_point = "%0" 
                if longer_tags == "1":
                    seller_point = ratingList[-1]   

            if longer_tags == "0":
                seller_pointList.append(seller_point[1:])
            else:
                seller_pointList.append(seller_point)

        elif store_name == "Trendyol":
            seller_point = ratingList[-1]
            seller_pointList.append(seller_point)
        #print(seller_pointList)
        #end of the seller name

        #print(productList)    
        price_classList.append(price_class)
        #end of the seller name
        #en dof the getdata function

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
    if len(productList) > 0 :
        print("first product name = ",productList[0])
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
    #nltk.download()
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
    man_woman_dict = {}
    brand_dict = {}
    sport_dict = {}
    color_dict = {}

    for item in  df_most_targetwords["data"]:     
        print(item)
        if counter<100:       
            table_dict_1[item[0]] = item[1]
        else:
            table_dict_2[item[0]] = item[1]

        if item[0] == "erkek" or item[0] == "bayan" or item[0] == "çocuk" or item[0] == "kız" or item[0] == "unisex":
             man_woman_dict[item[0]] = item[1]
        elif item[0] == "adidas" or item[0] == "nik" or item[0] == "pum" or item[0] == "nik" or  item[0] == "jump" or item[0] == "kinetix" or item[0] == "lumberjack" or item[0] == "slazenger":
             brand_dict[item[0]] = item[1]
        #brands continue
        elif "das" in item[0] or "net" in item[0]  or item[0] == "lescon" or item[0] == "lotto" or item[0] == "polo" or item[0] == "reebok" or item[0] == "hummel" or item[0] == "hammer"  or item[0] == "runfalcon":
             brand_dict[item[0]] = item[1]
        
        if item[0] == "futbol" or item[0] == "basketbol" or item[0] == "voleybol" or item[0] == "tenis" or item[0] == "koş" or item[0] == "yürüyüş"or item[0] == "futsal":
            sport_dict[item[0]] = item[1]
        
        if item[0] == "siyah" or item[0] == "beyaz" or item[0] == "gri" or item[0] == "lacivert" or item[0] == "haki" or item[0] == "mavi" or item[0] == "turunç" or item[0] == "yeşil":
            color_dict[item[0]] = item[1]           


        counter += 1

    #print(man_woman_dict)
    #print(brand_dict)
    # print(sport_dict)
    # print(color_dict)


    return table_dict_1, table_dict_2, man_woman_dict, brand_dict, sport_dict, color_dict


