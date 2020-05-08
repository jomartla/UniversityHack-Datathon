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

<img src="/png/unbalanced.png" width="200px">
We used SMOTE, a data augmentation technique to **downsample** the residencial class (selecting the most representative ones) and **oversample** the other classes. In the end, the class distribution looked like this:

<img src="/png/balanced.png" width="200px">

### Latitude and Longitude map:
The latitude and longitude values were displaced and scaled randomly, but conserving their relative postions.
<img src="/png/map.png" width="1000px">
### Correlation matrix:
We realized that the rgb & nir channel deciles were extremely correlated with each other. The exceptions to these rules were the 0th and 10th deciles from each channel, which had much lower correlation. 
<img src="/png/corr_matrix.png" width="1000px">

### RGB & NIR channels decile average:
To go further in our analysis of the channels we created this graph with the average value of each decile for each channel and terrain type
<img src="/png/channels.png" width="1000px">


## Model design

Explicar SMOTE data augmentation
Explicar XGboost

Explicar el modelo conjunto: Dense + CNN
