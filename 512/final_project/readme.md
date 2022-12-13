### Goal
We aim to assess the overall efficacy of mask mandates across the United States and examine if efficacy is associated with political affiliation of a region.

### Description
Initially, this repository contained only code to analyze the impace of mask mandate in Pierce County, WA. That research was expanded to include all counties in the United States. We additionally augmented the original datasets with county-level election results for the 2020 presidential election, which allowed us to approximate any given county's political lean (Democrat or Republican). We assess the overall efficacy of mask mandates across the entirety of the COVID-19 pandemic and test if there is any relationship with political lean of a region. 

Data comes from the US government, Harvard Databerse, and Johns Hopkins University via Kaggle. 

### Usage
Acquire the data from the below references, move it to this repositories datat directory (`./data`), and run the notebook. Everything is reproducible given the raw data. 

### Data
Data is acquired from the following internet sources:
* https://www.kaggle.com/datasets/antgoldbloom/covid19-data-from-john-hopkins-university
  * For our purposes, contains county-level COVID-19 daily case counts
* https://data.cdc.gov/Policy-Surveillance/U-S-State-and-Territorial-Public-Mask-Mandates-Fro/62d6-pm5i
  * For our purposes, contains county-level details of mask mandates, including begin and end dates
* https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/VOQCHQ 
  * For our purposes, contains the number of votes recieved by each presidential candidate in the 2020 election at the county-level 
