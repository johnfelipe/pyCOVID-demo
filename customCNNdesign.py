######### Show what the custom CNN coded design and corresponding model.summary() output

def blunder1():

    boulette = """## instanciation du modèle
    classifier = Sequential()
    
    ## construction des 7 couches
    conv1 = Conv2D(filters = 64,                 # couche composée de 32 matrices de convolution
                      input_shape = (224, 224, 1),  # l'image passée en entrée a 64 pixels de hauteur, 64 pixels de largeur et 3 canaux RGB
                      kernel_size = (3, 3),         # noyau de convolution de dimension 3x3 (9 neurones par filtre)
                      padding = "valid",            # pour que le noyau ne puisse pas dépasser les bords de l'image
                      activation = "relu")
    
    maxpool = MaxPooling2D(pool_size = (2, 2))    # le maximum sera calculé sur des tuiles de dimensions 2x2
    
    batchNorm = BatchNormalization()              # The layer will transform inputs so that they are standardized: mean = 0 and standard deviation = 1
    
    conv2 = Conv2D(filters = 64,                 # couche composée de 32 matrices de convolution
                      kernel_size = (3, 3),         # noyau de convolution de dimension 3x3 (9 neurones par filtre)
                      padding = "valid",            # pour que le noyau ne puisse pas dépasser les bords de l'image
                      activation = "relu")
    
    maxpool = MaxPooling2D(pool_size = (2, 2))    # test
    
    dropout = Dropout(rate=0.2)                  # test
    
    flatten = Flatten()                          # pour transformer les matrices en vecteurs à donner en input à des couches denses
    
    dense_1 = Dense(units = 1024,                  # nombre de neurones
                    activation ='relu')           # fonction d'activation
    
    dropout = Dropout(rate=0.2)
    
    dense_2 = Dense(units = 512,                  # nombre de neurones
                    activation ='relu')           # fonction d'activation
    
    dense_3 = Dense(units = n_class,                   # output layer
                    activation ='softmax')        # fonction d'activation
    
    ## ajout des couches au modèle
    classifier.add(conv1)
    classifier.add(maxpool)
    classifier.add(batchNorm)
    classifier.add(conv2)
    classifier.add(maxpool)
    # classifier.add(batchNorm)
    classifier.add(dropout)
    classifier.add(flatten)
    classifier.add(dense_1)
    classifier.add(dropout)
    classifier.add(dense_2)
    classifier.add(dropout)
    classifier.add(dense_3)"""

    return boulette

def blunder2():
    output = """
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_2 (Conv2D)            (None, 222, 222, 64)      640       
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 multiple                  0         
_________________________________________________________________
batch_normalization_1 (Batch (None, 111, 111, 64)      256       
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 109, 109, 64)      36928     
_________________________________________________________________
dropout_3 (Dropout)          multiple                  0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 186624)            0         
_________________________________________________________________
dense_2 (Dense)              (None, 1024)              191104000 
_________________________________________________________________
dense_3 (Dense)              (None, 512)               524800    
_________________________________________________________________
dense_4 (Dense)              (None, 3)                 1539      
=================================================================
Total params: 191,668,163
Trainable params: 191,668,035
Non-trainable params: 128
_________________________________________________________________"""

    return output
