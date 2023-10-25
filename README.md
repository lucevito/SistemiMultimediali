# output
    dataset paziente   foto1   foto2  ... Sesso Et? Data Nascita  buona
0       ita      T_1   92317   92247  ...     F  40       1979.0    NaN
1       ita      T_2   93707   93702  ...     F  34       1985.0    NaN
2       ita      T_3   94015   94012  ...     M  53       1966.0    NaN
3       ita      T_4   95329   95326  ...     M  36       1983.0    NaN
4       ita      T_5   74439   74432  ...     M  51       1968.0    NaN
..      ...      ...     ...     ...  ...   ...  ..          ...    ...
212     onc    O_10C  130959  130957  ...     M  59       1960.0    NaN
213     onc    O_11C   91410   91408  ...     M  22       1997.0    NaN
214     onc    O_12C   91505   91502  ...     F  76       1943.0    NaN
215     onc    O_13C   93333   93332  ...     F  70       1949.0    NaN
216     onc    O_14C  110116  110115  ...     F  54       1965.0    NaN

[217 rows x 10 columns]
working with: dataset\congiuntive\anemici
working with: dataset\congiuntive\non_anemici
working with: dataset\occhi\anemici
working with: dataset\occhi\non_anemici
libpng warning: iCCP: CRC error
libpng warning: iCCP: CRC error
libpng warning: iCCP: CRC error
libpng warning: iCCP: CRC error
libpng warning: iCCP: CRC error
libpng warning: iCCP: CRC error
libpng warning: iCCP: CRC error
libpng warning: iCCP: CRC error
libpng warning: iCCP: CRC error
libpng warning: iCCP: CRC error
libpng warning: iCCP: CRC error
libpng warning: iCCP: CRC error
libpng warning: iCCP: CRC error
libpng warning: iCCP: CRC error
libpng warning: iCCP: CRC error
libpng warning: iCCP: CRC error
libpng warning: iCCP: CRC error
libpng warning: iCCP: CRC error
libpng warning: iCCP: CRC error
libpng warning: iCCP: CRC error
libpng warning: iCCP: CRC error
libpng warning: iCCP: CRC error
libpng warning: iCCP: CRC error
libpng warning: iCCP: CRC error
libpng warning: iCCP: CRC error
libpng warning: iCCP: CRC error
libpng warning: iCCP: CRC error
libpng warning: iCCP: CRC error
libpng warning: iCCP: CRC error
libpng warning: iCCP: CRC error
libpng warning: iCCP: CRC error
libpng warning: iCCP: CRC error
libpng warning: iCCP: CRC error
libpng warning: iCCP: CRC error
libpng warning: iCCP: CRC error
libpng warning: iCCP: CRC error
libpng warning: iCCP: CRC error
libpng warning: iCCP: CRC error
libpng warning: iCCP: CRC error
libpng warning: iCCP: CRC error
libpng warning: iCCP: CRC error
libpng warning: iCCP: CRC error
libpng warning: iCCP: CRC error
libpng warning: iCCP: CRC error
libpng warning: iCCP: CRC error
libpng warning: iCCP: CRC error
libpng warning: iCCP: CRC error
libpng warning: iCCP: CRC error
libpng warning: iCCP: CRC error
libpng warning: iCCP: CRC error
libpng warning: iCCP: CRC error
libpng warning: iCCP: CRC error
libpng warning: iCCP: CRC error
libpng warning: iCCP: CRC error
libpng warning: iCCP: CRC error
libpng warning: iCCP: CRC error
libpng warning: iCCP: CRC error
libpng warning: iCCP: CRC error
libpng warning: iCCP: CRC error
libpng warning: iCCP: CRC error
libpng warning: iCCP: CRC error
libpng warning: iCCP: CRC error
Found 296 images belonging to 2 classes.
Found 74 images belonging to 2 classes.
2023-10-25 18:01:40.542772: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: SSE SSE2 SSE3 SSE4.1 SSE4.2 AVX AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 148, 148, 32)      896       
                                                                 
 max_pooling2d (MaxPooling2  (None, 74, 74, 32)        0         
 D)                                                              
                                                                 
 conv2d_1 (Conv2D)           (None, 72, 72, 64)        18496     
                                                                 
 max_pooling2d_1 (MaxPoolin  (None, 36, 36, 64)        0         
 g2D)                                                            
                                                                 
 conv2d_2 (Conv2D)           (None, 34, 34, 128)       73856     
                                                                 
 flatten (Flatten)           (None, 147968)            0         
                                                                 
 dense (Dense)               (None, 512)               75760128  
                                                                 
 dense_1 (Dense)             (None, 2)                 1026      
                                                                 
=================================================================
Total params: 75854402 (289.36 MB)
Trainable params: 75854402 (289.36 MB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
Epoch 1/20
19/19 [==============================] - 22s 1s/step - loss: 0.8072 - accuracy: 0.8649 - val_loss: 1.2114e-06 - val_accuracy: 1.0000
Epoch 2/20
19/19 [==============================] - 21s 1s/step - loss: 0.0011 - accuracy: 1.0000 - val_loss: 0.0000e+00 - val_accuracy: 1.0000
Epoch 3/20
19/19 [==============================] - 21s 1s/step - loss: 1.5308e-05 - accuracy: 1.0000 - val_loss: 1.2887e-08 - val_accuracy: 1.0000
Epoch 4/20
19/19 [==============================] - 21s 1s/step - loss: 4.5920e-05 - accuracy: 1.0000 - val_loss: 0.0000e+00 - val_accuracy: 1.0000
Epoch 5/20
19/19 [==============================] - 21s 1s/step - loss: 1.7478e-07 - accuracy: 1.0000 - val_loss: 0.0000e+00 - val_accuracy: 1.0000
Epoch 6/20
19/19 [==============================] - 21s 1s/step - loss: 5.7358e-04 - accuracy: 1.0000 - val_loss: 0.0000e+00 - val_accuracy: 1.0000
Epoch 7/20
19/19 [==============================] - 22s 1s/step - loss: 0.0000e+00 - accuracy: 1.0000 - val_loss: 0.0000e+00 - val_accuracy: 1.0000
Epoch 8/20
19/19 [==============================] - 23s 1s/step - loss: 0.0000e+00 - accuracy: 1.0000 - val_loss: 0.0000e+00 - val_accuracy: 1.0000
Epoch 9/20
19/19 [==============================] - 23s 1s/step - loss: 0.0000e+00 - accuracy: 1.0000 - val_loss: 0.0000e+00 - val_accuracy: 1.0000
Epoch 10/20
19/19 [==============================] - 23s 1s/step - loss: 0.0190 - accuracy: 0.9932 - val_loss: 6.2956e-04 - val_accuracy: 1.0000
Epoch 11/20
19/19 [==============================] - 25s 1s/step - loss: 1.7285e-04 - accuracy: 1.0000 - val_loss: 0.0000e+00 - val_accuracy: 1.0000
Epoch 12/20
19/19 [==============================] - 24s 1s/step - loss: 8.0547e-09 - accuracy: 1.0000 - val_loss: 0.0000e+00 - val_accuracy: 1.0000
Epoch 13/20
19/19 [==============================] - 25s 1s/step - loss: 4.0273e-09 - accuracy: 1.0000 - val_loss: 0.0000e+00 - val_accuracy: 1.0000
Epoch 14/20
19/19 [==============================] - 25s 1s/step - loss: 5.2355e-09 - accuracy: 1.0000 - val_loss: 0.0000e+00 - val_accuracy: 1.0000
Epoch 15/20
19/19 [==============================] - 25s 1s/step - loss: 9.2629e-09 - accuracy: 1.0000 - val_loss: 0.0000e+00 - val_accuracy: 1.0000
Epoch 16/20
19/19 [==============================] - 25s 1s/step - loss: 1.6109e-09 - accuracy: 1.0000 - val_loss: 0.0000e+00 - val_accuracy: 1.0000
Epoch 17/20
19/19 [==============================] - 25s 1s/step - loss: 8.0547e-10 - accuracy: 1.0000 - val_loss: 0.0000e+00 - val_accuracy: 1.0000
Epoch 18/20
19/19 [==============================] - 24s 1s/step - loss: 6.4437e-09 - accuracy: 1.0000 - val_loss: 0.0000e+00 - val_accuracy: 1.0000
Epoch 19/20
19/19 [==============================] - 24s 1s/step - loss: 2.4164e-09 - accuracy: 1.0000 - val_loss: 0.0000e+00 - val_accuracy: 1.0000
Epoch 20/20
19/19 [==============================] - 25s 1s/step - loss: 4.0273e-09 - accuracy: 1.0000 - val_loss: 0.0000e+00 - val_accuracy: 1.0000
37/37 [==============================] - 3s 72ms/step - loss: 0.0000e+00 - accuracy: 1.0000
[0.0, 1.0]
37/37 [==============================] - 3s 73ms/step - loss: 0.0000e+00 - accuracy: 1.0000

Process finished with exit code 0
