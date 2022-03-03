import json
from bs4 import BeautifulSoup
from django.shortcuts import get_object_or_404

from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
# pip install webdriver-manager

# data visualisition
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import seaborn as sns
import numpy as np
import requests
from .models import Post

def load_data(stock, look_back):
    data_raw = stock.values   # convert to numpy array
    print(type(data_raw))
    print(data_raw.shape)
    data = []
        # create all possible sequences of length look_back
    for index in range(len(data_raw) - look_back):
        data.append(data_raw[index: index + look_back])
    data = np.array(data)
    print(data.shape)
    test_set_size = int(np.round(0.2*data.shape[0]))
    train_set_size = data.shape[0] - (test_set_size)
    x_train = data[:train_set_size, :-1, :]
    y_train = data[:train_set_size, -1, :]
    x_test = data[train_set_size:, :-1]
    y_test = data[train_set_size:, -1, :]
    return [x_train, y_train, x_test, y_test]

def rescale(data, new_min, new_max, Y_coor):
        """Rescale the data to be within the range [new_min, new_max]"""
        return (data - min(Y_coor[1:])) / (max(Y_coor[1:]) - min(Y_coor[1:])) * (new_max - new_min) + new_min


def load_data_fut(stock, look_back):
        data_raw_2 = stock.values  # convert to numpy array
        print(type(data_raw_2))
        print(data_raw_2.shape)
        data_new = []
        
        # create all possible sequences of length look_back
        for index in range(look_back):
            data_new.append(data_raw_2[index: index + look_back])
        
        
        print("-----------------")
        print(data_new)
        data_new = np.array(data_new)
        print("shape = ",data_new.shape)
        
        return data_new



def get_time_series_data(request):

    browser = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    future_url_text = request.POST.get('future_text') # this is the name !!!
    print("future_url_text = ", future_url_text)
    browser.get(future_url_text)
    # browser.get("https://www.cimri.com/bebek-bezi/en-ucuz-sleepy-natural-no5-junior-24-adet-bebek-bezi-fiyatlari,293063002")

    html = None
    selector = '#main_container > div > div.s98wa6g-0.feTYBN > div.s98wa6g-3.iCxKVJ > div:nth-child(1) > div.s1fqyqkq-12.cbNHnS > div.s1fqyqkq-6.iWPMRj > div > div > svg > path'
    delay = 30  # seconds

    # browser = webdriver.Chrome()
    # browser.get(url)

    try:
        # wait for button to be enabled
        element = WebDriverWait(browser, delay).until(
            EC.element_to_be_clickable((By.XPATH, '//*[@id="main_container"]/div/div[1]/div[3]/div[1]/div[1]/div[1]/div/button[4]'))    
        )
        element.click()
        
        
        # wait for data to be loaded
        WebDriverWait(browser, delay).until(
            EC.presence_of_all_elements_located((By.CLASS_NAME, 'rv-xy-plot__inner'))
        
        ) 
        
    except TimeoutException:
        print('Loading took too much time!')
    else:
        html = browser.page_source
        print("html = ", html)
    finally:
        browser.quit()

    if html:
        soup = BeautifulSoup(html, 'lxml')
        raw_data = soup.select_one(selector)
        min_max_price = soup.select_one('#main_container > div.s1a29zcm-1.cmgeOC > div.s98wa6g-0.feTYBN > div.s98wa6g-3.iCxKVJ > div:nth-child(1) > div.s1fqyqkq-1.lmDmfb')
        
        # data = json.loads(raw_data)
        import pprint
        print("----datam-----")
        pprint.pprint(raw_data)
        print("values")


    # print(min_max_price)
    date = min_max_price.find_all("p", class_ ="s1fqyqkq-5 dekHBg")
    price = []
    for item in date:
        print(item.text.split())
        el = item.text.split()[0].replace(".","").replace(",",".")
        price.append(el)
        print(str(el))

    max_price = price[0] 
    min_price = price[1] 

    print("max_price = ",max_price, "min_price = ",min_price)

    element_all = raw_data["d"] 
    elements = raw_data["d"].split('L')
    print("elements =", elements)


    X_coor = [0]
    Y_coor = [0]
    for elem in elements[1:-2]:
        if float(elem.split(',')[0]) > float(X_coor[-1]):
            X_coor.append(float(elem.split(',')[0]))
            Y_coor.append(400-float(elem.split(',')[1]))

    print(Y_coor)
    print(min(Y_coor[1:]),"---",max(Y_coor))
    print(len(X_coor))
    
    # x axis values
    x = X_coor
    # corresponding y axis values
    y = Y_coor
    
    import matplotlib.pyplot as plt

    # plotting the points
    plt.plot(x, y)

    # plt.xlim(0,890) # 890
    # plt.ylim(63,112) # 280
    
    # naming the x axis
    plt.xlabel('x - axis')
    # naming the y axis
    plt.ylabel('y - axis')
    
    # giving a title to my graph
    plt.title('1 year graph!')
    
    # function to show the plot
    # plt.show()


    plt.style.use('ggplot')
    df = pd.DataFrame(y[1:])
    df.plot(label='1 Yıllık Fiyat', title='1 Yıllık Fiyat')
    

    print(df.head()) 
    print(df.tail())

    
    new_list = []
    for num in Y_coor[1:]:
        new_list.append(rescale(num, float(min_price), float(max_price), Y_coor))
                        
    print(new_list)   
    
    # x axis values
    x = X_coor[1:]
    # corresponding y axis values
    y = new_list
    
    # plotting the points
    plt.plot(x, y)

    # plt.xlim(0,890) # 890
    # plt.ylim(63,112) # 280
    # naming the x axis
    plt.xlabel('x - axis')
    # naming the y axis
    plt.ylabel('y - axis')
    
    # giving a title to my graph
    plt.title('1 year graph!')
    
    # function to show the plot
    # plt.show()
    

    df_new_y = pd.DataFrame(new_list)
    df_new_y.plot(figsize=(30, 6))
    plt.ylabel("Rescaled Product Price ")
    plt.title("1 Yıllık Fiyat")
    # plt.show()
    plt.savefig("./api/static/images/days_graph.png")

    # MinMaxScaler
    from sklearn.metrics import mean_squared_error
    from sklearn.preprocessing import MinMaxScaler

    df_scaled = df_new_y.copy()

    scaler = MinMaxScaler(feature_range=(-1, 1))
    df_scaled = scaler.fit_transform(df_new_y.values.reshape(-1, 1))
    df_scaled = pd.DataFrame(df_scaled)
    print(df_scaled)
    print(df_scaled.info())

    look_back = 30  # choose sequence length
    x_train, y_train, x_test, y_test = load_data(df_scaled, look_back)
    print('x_train.shape = ', x_train.shape)
    print('y_train.shape = ', y_train.shape)
    print('x_test.shape = ', x_test.shape)
    print('y_test.shape = ', y_test.shape)

    import torch
    # make training and test sets in torch
    x_train = torch.from_numpy(x_train).type(torch.Tensor)
    x_test = torch.from_numpy(x_test).type(torch.Tensor)
    y_train = torch.from_numpy(y_train).type(torch.Tensor)
    y_test = torch.from_numpy(y_test).type(torch.Tensor)

    # Build model
    input_dim = 1
    hidden_dim = 100
    num_layers = 2
    output_dim = 1

    model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
    loss_fn = torch.nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
    print(model)
    print(len(list(model.parameters())))
    
    for i in range(len(list(model.parameters()))):
        print(list(model.parameters())[i].size())

    # train the model
    num_epochs = 100
    hist = np.zeros(num_epochs)

    # Number of steps to unroll
    seq_dim = look_back-1

    for t in range(num_epochs):
        # Initialise hidden state
        # Don't do this if you want your LSTM to be stateful
        # model.hidden = model.init_hidden()
        # Forward pass
        y_train_pred = model(x_train)

        loss = loss_fn(y_train_pred, y_train)
        if t % 10 == 0 and t != 0:
            print("Epoch ", t, "MSE: ", loss.item())
            Post.objects.filter(id=1).update(title='% '+str(t))
            # duration images
        hist[t] = loss.item()

        # Zero out gradient, else they will accumulate between epochs
        optimiser.zero_grad()

        # Backward pass
        loss.backward()

        # Update parameters
        optimiser.step()


    plt.plot(hist, label="Training loss")
    plt.legend()
    # plt.show()



    import torch
    import math
    from torch.autograd import Variable

    # make predictions
    y_test_pred = model(x_test)


    # invert predictions
    y_train_pred = scaler.inverse_transform(y_train_pred.detach().numpy())
    y_train = scaler.inverse_transform(y_train.detach().numpy())
    y_test_pred = scaler.inverse_transform(y_test_pred.detach().numpy())
    y_test = scaler.inverse_transform(y_test.detach().numpy())

    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(y_train[:, 0], y_train_pred[:, 0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(y_test[:, 0], y_test_pred[:, 0]))
    print('Test Score: %.2f RMSE' % (testScore))



    from torch.autograd import Variable
    # Visualising the results
    figure, axes = plt.subplots(figsize=(15, 6))
    # axes.xaxis_index()

    axes.plot(df_new_y[len(df_new_y) - len(y_test):].index, y_test, color='red', label='Real Product Price')
    axes.plot(df_new_y[len(df_new_y) - len(y_test):].index, y_test_pred, color='blue', label='Predicted Product Price')

    # axes.xticks(np.arange(0,394,50))
    plt.title('Future Price Prediction')
    plt.xlabel('Time (Day)')
    plt.ylabel('Price (TL)')
    plt.legend()
    plt.savefig('price1.png')
    # plt.show()

    print("y_test_pred.shape = ", y_test_pred.shape)

    # https://gist.github.com/dcshapiro/7e72fa2e2a5a1b54bb8a5b911ea325b7
    # shift train predictions for plotting
    trainPredictPlot = np.empty_like(df_scaled[0:])
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(y_train_pred)+look_back, :] = y_train_pred

    # shift test predictions for plotting
    testPredictPlot = np.empty_like(df_scaled[0:])
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(y_train_pred)+(look_back):len(df_scaled), :] = y_test_pred

    # len(trainPredict)+(look_back*2)+1:len(dataset)-1, :

    # plot baseline and predictions
    figure, axes = plt.subplots(figsize=(15, 6))
    plt.plot(scaler.inverse_transform(df_scaled[0:]))
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.savefig("./api/static/images/future_actual_predicted.png")
    # plt.show()


    
    # future predict data processing
    fut_pred = 30

    test_inputs = df_scaled[-fut_pred*2:].values.tolist()
    print(test_inputs)


    model.eval()


    new_list = []
    for i in range(60):
        new_list.append(test_inputs[i][0])
    print(len(new_list))   

    look_back = 30  # choose sequence length

    df_list = pd.DataFrame (new_list)
    x_fut = load_data_fut(df_list, look_back)
    print('x_fut.shape = ', x_fut.shape)

    x_fut = torch.from_numpy(x_fut).type(torch.Tensor)

    y_pred_fut = model(x_fut)
    print('y_pred_fut.shape = ', y_pred_fut.shape)


    all_predictions = scaler.inverse_transform(y_pred_fut.detach().numpy().reshape(-1, 1))
    print(all_predictions)
    
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Visualising the results
    figure, axes = plt.subplots(figsize=(15, 6))
    # axes.xaxis_index()

    # axes.plot(df_new_y[-30:], color='red', label='Real Product Price')
    axes.plot(all_predictions, color='blue', label='Predicted Product Price')

    # axes.xticks(np.arange(0,394,50))
    plt.title('Future Price Prediction')
    plt.xlabel('Time (Day)')
    plt.ylabel('Price (TL)')
    plt.legend()
    plt.savefig('./api/static/images/price.png')
    # plt.show()


# Here we define our model as a class
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

            # Number of hidden layers
        self.num_layers = num_layers

            # batch_first=True causes input/output tensors to be of shape
            # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

            # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
            # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

            # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

            # We need to detach as we are doing truncated backpropagation through time (BPTT)
            # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

            # Index hidden state of last time step
            # out.size() --> 100, 32, 100
            # out[:, -1, :] --> 100, 100 --> just want last time step hidden states!
        out = self.fc(out[:, -1, :])
            # out.size() --> 100, 10
        return out
        
