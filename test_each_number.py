
import cv2
import numpy as np
import pickle
from keras.models import load_model

windowName = 'Drawing Demo'
img = np.zeros((224, 224, 3), np.uint8)
cv2.namedWindow(windowName)

# true if mouse is pressed
drawing = False

(ix, iy) = (-1, -1)

# mouse callback function
def draw_shape(event, x, y, flags, param):
    global ix, iy, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        (ix, iy) = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.circle(img, (x, y), 6, (255, 255, 255), -1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.circle(img, (x, y), 6, (255, 255, 255), -1)

cv2.setMouseCallback(windowName, draw_shape)

def testing_sigle_value(temp,weights,biases):
    temp = np.asarray(temp)
    np.reshape(temp,(784,1))
    Z1_temp = np.dot(weights['w1'].T,temp)+biases['b1']#256*784
    Layer1_temp = np.maximum(Z1_temp,0)
    Z2_temp = np.dot(weights['w2'].T,Layer1_temp)+biases['b2']#256*1
    Layer2_temp = np.maximum(Z2_temp,0)
    Zout_temp = np.dot(weights['out'].T,Layer2_temp)+biases['out'] #10*1
    predict_class = np.argmax(Zout_temp,axis=0)

    precen = softmax(Zout_temp)
    #k_temp = np.reshape(precen,(,10))
    ind = np.argpartition(precen.T, -3)[-3:]
    
  
    model = load_model('MNIST_train.h5')
    ff = model.predict(temp.reshape(1,28,28,1))
    print("=======================================================")
    print("CNN Prediction result  : ",np.argmax(ff,axis=1))
    print("MLP prediction result : ",predict_class)
    #return predict_class		
    # print("prediction result : ",predict_class,"bbbbb",ind[0])
    # print("prediction result : ",predict_class,"bbbbb",ind[0][8])
    # print("prediction result : ",predict_class,"bbbbb",ind[0][7])


def load_NN():
    '''
    Load Neural NetWork from file
    '''
    pickle_read = open("data.pkl","rb")
    data = pickle.load(pickle_read)
    weights = data['weights']
    biases = data['biases']
    return weights,biases

def softmax(V):
    # tru cho np.max de chống tràn số khi số mũ quá lớn
    e_V = np.exp(V - np.max(V, axis = 0, keepdims = True))
    Z = e_V / e_V.sum(axis = 0)
    return Z

def main():
    weights,biases = load_NN()
    while(True):
        cv2.imshow(windowName, img)
        if cv2.waitKey(1) == 27: break

    #cv2.destroyAllWindows()
    temp_img_1 = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    temp_img_2 = cv2.resize(temp_img_1,(28,28))
    #cv2.imshow(windowName,temp_img_2)
    #print(temp_img_2)
    temp_img_3 = np.reshape(temp_img_2,(784,1))
    testing_sigle_value(temp_img_3,weights,biases)
    #print("prediction result is :",t)
    #print()
    #print(temp_img_2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()