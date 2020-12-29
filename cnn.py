#part 1 Building the convolutional neural network
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
'''Sequential is to initialise the neural network
Convolution2D is the package to make the convolution step in cnn
as here we are working with images so this is 2D
Maxpooling2D to add pooling layers
Flatten to convert all max pooled layers into one layer input
Dense is to add fully connected layer 
'''
#CREATING THE MODEL---------
#initialize the convolutional neural network
classifier=Sequential()

#Step 1-Convolution

'''Adding 32 feature detector and 3X3 matrix Suppose the input size is 3,256,256
so as the picture is a colored image so there will be 3 channels with size 256,256
this input shape=3.256,256 is a theano backend but we are using tensorflow backend
so the order will be changed input shape=(256,256,3)'''
classifier.add(Convolution2D(32,3,3,input_shape=(64,64,3),activation='relu'))

#Step 2 Pooling   

#to use the reduction of size of the feature map or the nodes
classifier.add(MaxPooling2D(pool_size=(2,2)))
#this line will reduce the feature map by 2

#adding second convolutional layer
classifier.add(Convolution2D(32,3,3,input_shape=(64,64,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

#adding third convolutional layer
classifier.add(Convolution2D(64,3,3,input_shape=(64,64,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

#Step 3 Flattening

classifier.add(Flatten())

#Step 4 Full Connection

#Hidden layer
classifier.add(Dense(units=128,activation='relu'))

#Output Layer
classifier.add(Dense(units=1,activation='sigmoid'))

#Compiling the cnn
'''as we have binary outcome as there are only dog and cat we will choose binary cross
entropy ,if we have more than two we need to choose categorical cross entropy'''
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#IMAGE PREPROCESSING-------------
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory(
        'train',
        target_size=(64,64),
        batch_size=32,
        class_mode='binary')
test_set= test_datagen.flow_from_directory(
        'test',
        target_size=(64,64),
        batch_size=32,
        class_mode='binary')
classifier.fit_generator(
        training_set,
        steps_per_epoch=5996,
        epochs=10,
        validation_data=test_set,
        validation_steps=1000)
