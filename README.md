# MADS-2_FinancialManagementGroupProject
## Do CEOs impact firm performance?

A project by Jasmin Capka, Irene Iype, MyThu Lam, Tajda Urankar, Carla Weidner

This repository has the following structure:
```
|
|-- data                                                    Data csv files after preparation
|   |-- CCM_Fundamentals_Annual_2006_-_2021_normal.csv
|   |-- CCM_Fundamentals_Annual_2006_-_2021_winsorized.csv
|   |-- Compustat_annual_return.csv
|   |-- CRSP_annual_standard_deviation.csv
|   |-- Execucomp_2006_-_2021.csv
|-- data_preparation                                        Files preparing the data to be used
|   |-- additional_attributes.py
|   |-- compustat_annual_return.py
|   |-- Compustat_windsoring.py
|   |-- CRSP_StandardDeviation.py
|-- results                                                 Files with resulting models
|   |-- result_final_model.txt
|   |-- result_first_model.txt
|   |-- result_second_model.txt
|-- ceo_project.ipnb                                        Main workbook containing the models
|-- README.md                                               Short info 
```

In order to reproduce our results, please create the new folder ```data_raw``` and put the externally provided files into the ```data_raw``` folder.
Then run all the files in the ```data_preparation``` folder.
The prepared data files are then stored in the ```data``` folder.
After that, you can run the ```ceo_project.ipynb```.
The models' results are additionally manually stored in the ```results``` folder.