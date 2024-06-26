# Library Versions
# The following code was developed using the following libraries  
# tensorflow = 1.4
# keras = 2.1.5
# sklearn = 0.22.2
# matplotlib = 3.0.3


import os
from sklearn.metrics import confusion_matrix
import pandas as pd
from keras.utils.vis_utils import plot_model
from keras.callbacks import ModelCheckpoint
from matplotlib import pyplot as plt
import cv2
from sklearn.utils import shuffle
from keras.utils import np_utils
import csv
from keras.layers import *
from keras import *
import pathlib
from keras.applications.resnet50 import ResNet50
from sklearn.utils import class_weight



# Setting GPU ID
os.environ["CUDA_VISIBLE_DEVICES"]="0"
BasePath = os.getcwd()
print("BASE PATH : ",BasePath)

#Parameters
num_class = 4
epochs = 20
batch_size = 8
ImgSize = 224
learn = 0.001
Experiment_Name = 'ResNet50-10-Fold-Only_Images'

#Clinical Information Model
def create_mlp(dim):
	# define our MLP network
    model = Sequential()
    model.add(Dense(32, input_dim=dim, activation="relu"))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(12, activation="relu"))
    return model
# Costum Data Generator for the multi Input
def load_samples(csv_file):
    print()
    data = pd.read_csv(csv_file)
    data = data[['FileName', 'Label', 'ClassName']]
    file_names = list(data.iloc[:,0])
    # Get the labels present in the second column
    labels = list(data.iloc[:,1])
    samples=[]
    for samp,lab in zip(file_names,labels):
        samples.append([samp,lab])
    return samples
def shuffle_data(data):
    data = shuffle(data)
    return data
def preprocessing(img,label):
    img = cv2.resize(img,(ImgSize,ImgSize))
    img = img/255
    label = np_utils.to_categorical(label, 4)
    #print(label)
    return img,label
def preprocessing2(img,label):
    img = cv2.resize(img,(ImgSize,ImgSize))
    img = img/255
    label = np_utils.to_categorical(label-1, 4)
    #print(label)
    return img,label
def data_generator(samples, samples2, batch_size, shuffle_data=True, resize=224):
    num_samples = len(samples)
    print("THE LENGTH = ", num_samples)
    while True:  # Loop forever so the generator never terminates
        #samples = shuffle(samples)
        # Get index to start each batch: [0, batch_size, 2*batch_size, ..., max multiple of batch_size <= num_samples]
        for offset in range(0, num_samples, batch_size):
            # Get the samples you'll use in this batch
            batch_samples = samples[offset:offset + batch_size]
            batch_samples2 = samples2[offset:offset + batch_size]
            # Initialise X_train and y_train arrays for this batch
            X_train = []
            y_train = []
            PX_train = []
            Py_train = []
            X2_train = []
            Y2_train = []
            # For each example

            # CDI Data
            for batch_sample in batch_samples:
                # Load image (X) and label (y)
                img_name = batch_sample[0]
                label = batch_sample[1]
                #print(img_name)
                img = cv2.imread(os.path.join(img_name))
                img, label = preprocessing(img, label)
                # Add example to arrays
                X_train.append(img)
                y_train.append(label)
                X2_train.append(img)
                Y2_train.append(label)

            # Pneumonia Data
            for batch_sample in batch_samples2:
                # Load image (X) and label (y)
                img_name = batch_sample[0]
                label = batch_sample[1]
                #print(img_name)
                img = cv2.imread(os.path.join(img_name))
                img, label = preprocessing2(img, label)
                # Add example to arrays
                PX_train.append(img)
                Py_train.append(label)
                X2_train.append(img)
                Y2_train.append(label)


            # Make sure they're numpy arrays (as opposed to lists)
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            PX_train = np.array(PX_train)
            Py_train = np.array(Py_train)

            X2_train = np.array(X2_train)
            #print("The Shape of the attay is : ", y_train.shape)
            Y2_train = np.array(Y2_train)
            #empty = np.empty([8,4])
            #Py_train = np.concatenate((empty, Py_train), axis=0)
            #y_train = np.concatenate((y_train, empty), axis=0)
            #print("The Shape of the attay is : ", y_train.shape)
            yield [X_train,PX_train], [y_train,Py_train]

# Model Training
def Train_Model(files):

    train_data_path = files + '/Train.csv'
    Validate_data_path = files + '/Test.csv'
    test_data_path = files + '/Test.csv'

    img_shape = (ImgSize, ImgSize, 3)
    img_shape2 = (ImgSize,ImgSize,6)
    inputs = Input(img_shape)
    inputs2 = Input(img_shape)
    #JoinedInputs = add([inputs,inputs2])
    BaseModel = ResNet50(include_top=False, weights=None, input_shape=img_shape, pooling=None)
    BaseModel.trainable = True

    baseMOdel1 = BaseModel(inputs)
    baseMOdel2 = BaseModel(inputs2)

    addlayer = concatenate([baseMOdel1,baseMOdel2])
    #baseMOdel = BaseModel(outputs1)
    outputs = BatchNormalization()(addlayer)
    qoutputs = BatchNormalization()(outputs)
    qGoutputs = GlobalAveragePooling2D()(qoutputs)
    #outputs = Dropout(0.5)(qGoutputs)
    outputs = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001))(qGoutputs)
    outputs = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001))(outputs)

    #outputs = Dropout(0.5)(outputs)
    x = Dense(num_class, activation='softmax', name = 'CDI')(outputs)  # Only Images
    y = Dense(num_class, activation='softmax', name = 'Pneumonia')(outputs)  # Only Images



    # ***************************************************************************************

    model = Model(inputs=[inputs,inputs2], outputs=[x,y])
    sgd = optimizers.SGD(lr=learn, nesterov=True)
    losses = {
        "CDI": "binary_crossentropy",
        "Pneumonia": "binary_crossentropy",
    }
    lossWeights = {"CDI": 1.0, "Pneumonia": 1.0}

    model.compile(loss=losses,

                  optimizer=sgd,
                  metrics=['accuracy'])

    # Plotting the Model Architecture
    SaveImgPath = BasePath+ "/Saved_Images"
    file_Image = pathlib.Path(SaveImgPath)
    if file_Image.exists():
        print("Saved Images Path : ", SaveImgPath)
    else:
        os.makedirs(SaveImgPath)
        print("Saved Images Path : ", SaveImgPath)
    plot_model(model, SaveImgPath + "/" + Experiment_Name + "_model.png", show_shapes=True)


    # Save Path for Wieghts Checkpoints
    SaveDirPath = BasePath+ "/Saved_Model"
    file_weights = pathlib.Path(SaveDirPath)
    if file_weights.exists():
        print("Weights Path : ",SaveDirPath)
    else:
        os.makedirs(SaveDirPath)
        print("Weights Path : ",SaveDirPath)
    Checkpoint_Weights_path = SaveDirPath + '/CheckPointModel.h5'
    checkpoint = ModelCheckpoint(Checkpoint_Weights_path, monitor='val_dense_2_acc', verbose=0, save_best_only=True, mode='auto')
    callbacks_list = [checkpoint]

    train_samples = load_samples(train_data_path)
    Val_samples = load_samples(Validate_data_path)

    #for PneumoniaData
    train_data_path2 = '/home/mohamad/Desktop/Dual_OutPUT/Data/data_files_Pneumonia/Train.csv'
    Validate_data_path2 = '/home/mohamad/Desktop/Dual_OutPUT/Data/data_files_Pneumonia/Test.csv'
    train_samples2 = load_samples(train_data_path2)
    Val_samples2 = load_samples(Validate_data_path2)


    num_train_samples = len(train_samples)
    num_Val_samples = len(Val_samples)
    print('number of train samples: ', num_train_samples)
    print('number of Validation samples: ', num_Val_samples)



    # Create generator
    train_generatorCustom = data_generator(train_samples, train_samples2, batch_size=batch_size)
    validation_generatorCustom = data_generator(Val_samples, Val_samples2, batch_size=batch_size)
    STEP_SIZE_TRAIN = num_train_samples / batch_size
    STEP_SIZE_VALID = num_Val_samples / batch_size
    # Train the Model

    history = model.fit_generator(train_generatorCustom,
                                  steps_per_epoch=STEP_SIZE_TRAIN,
                                  epochs=epochs, validation_data=validation_generatorCustom,
                                  validation_steps=STEP_SIZE_VALID,
                                  callbacks=callbacks_list)
    print("Training Complete and Weights are saved")
    # Saving the Model
    Save_Model_Path = SaveDirPath + "/Complete_Model_Weights.h5"
    model.save(Save_Model_Path)


    # Plotting the Learning Curves
    # Accuracy
    for key in history.history.keys():
        print(key)
    plt.plot(history.history['CDI_acc'])
    plt.plot(history.history['val_CDI_acc'])
    plt.title('Model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    # plt.show()
    plt.savefig(SaveImgPath + '/Model_Accuracy_' + Experiment_Name + '.png')
    # Loss
    plt.plot(history.history['val_loss'])
    plt.plot(history.history['val_CDI_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    # plt.show()
    plt.savefig(SaveImgPath + '/Model_Loss_' + Experiment_Name + '.png')



    # ***************************************************
    # Testing the Saved Model
    # ***************************************************
    test_samples = load_samples(test_data_path)
    test_samples2 = load_samples(test_data_path)

    num_test_samples = len(test_samples)
    print('number of Test samples: ', num_test_samples)
    STEP_SIZE_Test = num_test_samples / batch_size
    Test_generatorCustom = data_generator(test_samples, test_samples2, batch_size=batch_size)
    print('Loading Saved Weights')
    model.load_weights(Checkpoint_Weights_path)

    # Confution Matrix and Classification Report
    print("Prediction Started")
    Y_pred = model.predict_generator(Test_generatorCustom, steps=STEP_SIZE_Test)
    y_pred = np.argmax(Y_pred, axis=1)

    print('Confusion Matrix')
    # X_Val, y_val = next(iter(validation_generatorCustom))

'''
    # Reading Ground Truths for comparision with prediction's
    df = pd.read_csv(test_data_path, usecols=['Label']) 
    y_true = sum(df.values.tolist(), [])

    array = confusion_matrix(y_true, y_pred)
    print(Experiment_Name)
    print(array)
    TP = array[0][0]
    FP = array[0][1]
    FN = array[1][0]
    TN = array[1][1]
    # print(array)
    print(" TP : " + TP.__str__() + ", TN : " + TN.__str__() + ", FN : " + FN.__str__() + ", FP : " + FP.__str__())
    # Saving the Result in a Text File

    ResultPath = BasePath + "/Results"
    file_Result = pathlib.Path(ResultPath)
    if file_Result.exists():
        print("Result Path : ", ResultPath)
    else:
        os.makedirs(ResultPath)
        print("Result Path : ", ResultPath)
    f = open( ResultPath + "/" +Experiment_Name, "a" )
    f.write(" TP " + TP.__str__() + " TN " + TN.__str__() + " FN " + FN.__str__() + " FP " + FP.__str__())
    f.close()
    # print("Accuracy : ", accuracy_score(y_true, y_pred))
    percision = TP / (TP + FP)
    recall = TP / (TP + FN)
    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    print("Percision : ", round(percision, 4))
    print("Recall : ", round(recall, 4))
    print("Accuracy : ", round(Accuracy, 4))

'''


if __name__=='__main__':
    Data_file = BasePath + "/Data/data_files"
    Train_Model(Data_file)
