Describe the goal of the project.
List the license of the source data and a link to the Wikimedia Foundation REST API terms of use: https://www.mediawiki.org/wiki/REST_API#Terms_and_conditions
Link to all relevant API documentation
Clearly name any intermediary data files and any final output files that your code creates. And for any files that your code creates you should describe the values of all fields.
List any known issues or special considerations with the data that would be useful for another researcher to know. 

### Goal
Lightly analyze viewing tends for Wikipedia articles on dinosaurs.

### Description
We use Wikipedia's web API service for acquire the data. License: https://www.mediawiki.org/wiki/REST_API#Terms_and_conditions

### Usage
Simply execute the IPython notebook contained in this repository from top to bottom.

### Data files
Data is downloaded directly from https://www.mediawiki.org/wiki/REST_API. We are concerned with three access types: mobile_
All data resides within `data` directory. All data transformations occur in memeory and are documented within the code. 
```
dino_monthy_cumulative_20150101002022093000.json
dino_monthy_desktop_20150101002022093000.json
dino_monthy_mobile_20150101002022093000.json
```
