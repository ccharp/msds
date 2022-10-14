### Goal
Examine wikipedia article rating bias via per-capita, geograhic analysis of politician articles.

### Usage
Simply execute the IPython notebook contained in this repository from top to bottom.

### Data files
The data required to bootstrap the analysis and acquire article quality was given in CSV form by the class instructor. URLs are evident in the code below. I do not know the lifespan of these URLs or how how they were gathered.

Some notable countries are inexplicably missing from the politician data, e.g. United States and United Kingdom, as well as a slew of very low population countries. The full list is presented in in the analysis section of this document. 

Intermediate datasets are stored for each step: acquisition, pre-processing, and analysis.

#### Sources
https://en.wikipedia.org/wiki/Category:Politicians_by_nationality
https://docs.google.com/spreadsheets/u/0/d/1Y4vSTYENgNE5KltqKZqnRQQBQZN5c8uKbSM4QTt8QGg/edit

https://www.prb.org/international/indicator/population/table/
https://docs.google.com/spreadsheets/u/0/d/1POuZDfA1sRooBq9e1RNukxyzHZZ-nQ2r6H5NcXhsMPU/edit

#### Local data
The result of performing transofmrations on data acquired at the above URLs. See `experiment.ipynb` for specifics. 
```
politicians_by_country_raw.csv
population_by_country_raw.csv
combined.csv
```

### Research Implications
The "per capita" constraint certainly makes things interesting. I, perhaps naively, expected per capita quality to increase with population size, yet the actual pattern is not clear, at least at the country level. For example, China, India, and Romania are among the countires with lowest number of articles per capita. However, Romania's articles tend to be much higher quality than the global average, while India's and China's do not. However, we do observe some overlap between the lists for high number of articles per capita and high article quality per capita: Andorra and Montenegro

Given that we used English Wikipedia as a data source, I am not surprised that the region that birthed English, SOUTHERN EUROPE, dominates the quality index. The reason for this bias is unclear, though I speculate that articles written for regions containing high prevelance of natvie English speakers typically rate higher than articles written elsewhere. 

To test this claim, I'd like to compare ORES score per capita with per capita English ability. It could also be elucidating to perform this same analysis for non-English wikis. 