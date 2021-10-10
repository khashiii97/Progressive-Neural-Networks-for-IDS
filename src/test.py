# This file is for testing the PNN
# you will need to make some changes since the data isn't available


import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import time
import matplotlib.pyplot as plt
import src.config as config
from src.Column_Genearator import *
from src.ProgNet import ProgNet


# since the data lies on our server in the INL,these two lines need to be implemented by anyone who want's to use this code.
from data import DataController
from reporter import Reporter



# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def evaluate_presicion(model,X,y,class_type = 'binary',interval_length = 100):
    tp = 0
    fp = 0
    fn = 0
    corrects = 0
    #if we handout all the inputs to the model at once, we will probably run in to a memory problem
    start_index = 0 #starting index of the interval of x we want to predict
    length = len(X)
    while start_index + interval_length < length:
        predicted_output = model(model.numCols - 1, X[start_index : start_index + interval_length])[:, -1]
        predicted_labels = torch.argmax(predicted_output, dim=1)
        if class_type == 'binary':
            for i in range(start_index,start_index + interval_length):
                if y[i] == predicted_labels[i - start_index]:
                    corrects += 1
        start_index += interval_length
    precision = corrects / len(y)
    return precision
controller = DataController(batch_size=config.column_batch_size,mode='binary',flatten=False,data_list=config.all_labels)
column_generator = Column_generator_LSTM(input_size=config.pkt_size,hidden_size=128,num_dens_Layer=2,num_LSTM_layer=1,num_of_classes=2)
binary_pnn = ProgNet(column_generator)
binary_pnn.to(device)
# create a validation set of 2500 flows
X_val = []
y_val = []
while 1:
    data = controller.generate('validation')
    if data == False:
      break
    X_val += data['x'].tolist()
    y_val += data['y'].tolist()
print("number of validation flows: ", len(X_val))

X_val = torch.tensor(np.array(X_val)).float().to(device)
y_val = np.array(y_val)


loss_criteria = nn.BCELoss()
precisions = []
number_of_columns = 0
while 1:
    start = time.time()
    data = controller.generate('train',output_model='prob')
    if data is False:
        break
    idx = binary_pnn.addColumn(device=device)
    flows = data['x'] # the 100 flows we will use to train a new column
    labels = data['y']
    optimizer = optim.Adam(binary_pnn.parameters(), lr=config.learning_rate)
    for epoch in range(config.epochs):
        current_flow_index = 0
        while current_flow_index  < len(flows):
            # training on batch
            optimizer.zero_grad()
            X_train = Variable(torch.tensor(flows[current_flow_index:current_flow_index + config.learning_batch_size]).float().to(device))
            y_train = Variable(torch.tensor(labels[current_flow_index:current_flow_index + config.learning_batch_size]).float().to(device))
            output =  binary_pnn(idx,X_train)[:,-1]
            loss = loss_criteria(output.float(),y_train)
            loss.backward()
            optimizer.step()
            current_flow_index += config.learning_batch_size

    # training of current column has ended
    # it weights shouldn't be modified anymore
    binary_pnn.freezeAllColumns()
    if number_of_columns % 1 == 0 :
        precision = evaluate_presicion(binary_pnn,X_val,y_val)
        precisions.append(precision)
        print("columns created : ", number_of_columns + 1)
        print("precision : ", precision)
    number_of_columns += 1
    end = time.time()
    print(str(end - start) ," seconds")

torch.save(binary_pnn.state_dict(),'binary-PNN-weights.pth')
plt.plot([i * 10 for i in range(number_of_columns)], precisions, color='b')
plt.suptitle('precisions')
plt.show()
# create a test set of 1000 flows
X_test = []
y_test = []
while 1:
    data = controller.generate('test')
    if data is False:
        break
    X_test += data['x'].tolist()
    y_test += data['y'].tolist()
X_test = torch.tensor(np.array(X_test)).float().to(device)
y_test = np.array(y_test)

final_precision = evaluate_presicion(binary_pnn,X_test,y_test)
print("Precision on test: ",final_precision)




