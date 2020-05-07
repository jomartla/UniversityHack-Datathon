# UniversityHack 2020 - Minsait Challenge
*Authors: [@jomartla](https://github.com/jomartla) &amp; [@MateuszKlimas](https://github.com/MateuszKlimas) &amp; [@juanluisrto](https://github.com/juanluisrto)*

This repository contains our solution for [the Minsait Challenge](https://www.cajamardatalab.com/datathon-cajamar-universityhack-2020/retos/predictivo/), a datathon organized by Cajamar UniversityHack.
We made it to the final round ([diplomas](/diplomas/)), creating a model which yielded an accuracy of **61%**.

We provide here a short summary of the problem, our findings and the design of our predictive model. You can also read the [final report](final_report.pdf) (in spanish) and take a look at the [notebook](Code/MINSAIT_bester_manns_final_solution.ipynb) with all the code.

## Problem description
The task at hand consisted in classifying terrain parcels among 7 different types.

We were given data about satellite pictures of parcels in Madrid. From each picture, we got the decile distribution of the **RGB & NIR colour channels**, as well as other information like the parcel's coordinates (relative to the rest of the parcels), the year of construction, geometric properties of the parcel, among other. The target variable was one of the following parcel types: **`[AGRICULTURE, INDUSTRIAL, OFFICE, OTHER, PUBLIC, RESIDENTIAL, RETAIL]`**

## Data exploration
We began exploring the dataset, and soon realized 

- Imagen distribución desequilibrada de los datos
- Imagen coordenadas
- Imagen rgb channels
- Imagen matrix correlacion


## Model design

Explicar SMOTE data augmentation
Explicar XGboost

Explicar el modelo conjunto: Dense + LSTM
