# UniversityHack 2020 - Minsait Challenge
*Authors: [@jomartla](https://github.com/jomartla) &amp; [@MateuszKlimas](https://github.com/MateuszKlimas) &amp; [@juanluisrto](https://github.com/juanluisrto)*

This repository contains our solution for [the Minsait Challenge](https://www.cajamardatalab.com/datathon-cajamar-universityhack-2020/retos/predictivo/), a datathon organized by Cajamar UniversityHack.
We made it to the final round ([diplomas](/diplomas/)), creating a model which yielded an accuracy of **61%**.

We provide here a short summary of the problem, our findings and the design of our predictive model. You can also read the [final report](final_report.pdf) (in spanish) and take a look at the [notebook](Code/MINSAIT_bester_manns_final_solution.ipynb) with all the code.

## Problem description
The task at hand consisted in classifying terrain parcels among 7 different types.

We were given data about satellite pictures of parcels in Madrid. From each picture, we got the decile distribution of the **RGB & NIR colour channels**, as well as other information like the parcel's coordinates (relative to the rest of the parcels), the year of construction, geometric properties of the parcel, among other. The target variable was one of the following parcel types: **`[AGRICULTURE, INDUSTRIAL, OFFICE, OTHER, PUBLIC, RESIDENTIAL, RETAIL]`**

## Data exploration & augmentation
We began exploring the dataset, and soon realized that classes were extremely unbalanced. We knew that the values we had to estimate would be much more balanced, so training a model with a more balanced dataset was from the beggining a priority.

<img src="/png/unbalanced.png" width="300px">

We used SMOTE, a data augmentation technique to **downsample** the residencial class (selecting the most representative ones) and **oversample** the other classes. In the end, the class distribution looked like this:

<img src="/png/balanced.png" width="300px">

### Latitude and Longitude map:
The latitude and longitude values were displaced and scaled randomly, but conserving their relative postions. We could appreciate that there exist clusters of parcels of the same kind, specially in the outskirts of Madrid. But most of the datapoints are completely blended. 
<img src="/png/map.png" width="850px">
### Correlation matrix:
We realized that the rgb & nir channel deciles were extremely correlated with each other. The exceptions to these rules were the 0th and 10th deciles from each channel, which had much lower correlation. 
<img src="/png/corr_matrix.png" width="1000px">

### RGB & NIR channels decile average:
To go further in our analysis of the channels we created this graph with the average value of the deciles for each channel and parcel type. We could notice substantial differences in the shape and range of the decile distribution among the 4 channels. 
<img src="/png/channels.png" width="1000px">



## Model design

The principal problem was trying to distinguish the RESIDENTIAL class from the rest of the classes. We tried to apply the XGboost algorithm as a first filter, in order to pass the entries classified as non-residential to a second model. XGBoost did not perform as expected so we finally dropped this idea.

In the end, we designed a model which treats differently the RGB & NIR channel features from the rest (construction year, maxbuilding floor, etc...). 
* The colour distributions can be considered as series, and to learn these series locality we input them to a 1 dimensional CNN
* The rest of the features are passed to a dense net

Both parts of the model are concatenated in the end and trained together.

```python
def merged_net():
    
    #DENSE net
    dense_input = Input(shape=(10,), name='main_input')
    dense = Dense(250, activation='relu')(dense_input)
    dense = Dropout(0.2)(dense)
    dense = BatchNormalization()(dense)
    dense = Dense(200, activation='relu')(dense)
    dense = Dropout(0.2)(dense)
    dense = BatchNormalization()(dense)
    dense= Dense(100, activation='relu')(dense)
    dense = Dropout(0.2)(dense)
    dense = BatchNormalization()(dense)
    dense = Dense(100, activation='relu')(dense)
    dense = Dropout(0.2)(dense)
    dense_out = BatchNormalization()(dense)
    
    #CNN1D
    seq_length = 11
    kernel = 2
    
    cnn_input = Input(shape=(seq_length,4), name='cnn_input')
    cnn = Conv1D(32, kernel_size= kernel, activation='relu')(cnn_input)
    cnn = Conv1D(64, kernel_size= kernel, activation='relu')(cnn)
    cnn = Conv1D(64, kernel_size= kernel, activation='relu')(cnn)
    cnn = Conv1D(128, kernel_size= kernel, activation='relu')(cnn)
    cnn = Flatten()(cnn)
    cnn_out = Dropout(dropout_rate)(cnn)

    #MERGED net
    concat = concatenate([cnn_out, dense_out])
    concat = Dense(50, activation='relu')(concat)
    model_output = Dense(7, activation='sigmoid')(concat)
    
    model = Model(inputs=[cnn_input, dense_input], outputs=[model_output])

    model.compile(loss='categorical_crossentropy',
                  optimizer= "nadam" ,
                  metrics=["accuracy"])
    return model
```
