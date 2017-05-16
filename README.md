# ny_phil_selection

This README will temporarily serve as a historical explanation of the steps taken in the evolution of this project.

# Initial Concept

The goal of this project is to track with time the decisions made in compositions and composers as functions of the following features: place of origin of composer, conductor, soloist instruments, and historical events that may have influenced these choices such as avoiding composers from regions at war with the US.

Currently I am using data from https://github.com/nyphilarchive/PerformanceHistory . The basic_eda.py file represents some of the fundamental EDA an reorganization I am performing in order to shape the data into a workable dataset that can be analyzed with standard regression methods and other custom modeling tools.  

# Further Exploration

After initial data exploration, I have added outside data collected from the New York Coincident Economic Indicators data located here.

https://www.newyorkfed.org/research/regional_economy/coincident_summary.html

With these as labels we hope to be able to find the indirect relationship between external economic factors on the choice of programs.

I have engineered some additional features from the data and continue to do so. After some initial linear regression scoring, which has revealed a very low correlation between the number of most frequent composers and CEI data, more relevant features such as those indicating smaller orchestras (eg: keywords of CONCERTO and QUARTET). I will be looking for additional economic data and budgetary information.

# Project Redirect

The ultimate goal of this project was to evaluate the choices made for the NYPhil based on external economic influences. As such, I am reshaping the data to create a more useful label that evaluates a program by unconventionality of the program works. Penalized are composers and pieces that regularly appear. This processed value will be indexed by date of the first concert for the program. This will be evaluated with exogenous featurized economics data pulled from various sources that will be listed here later once I have chosen the most relevant datasets. Included so far are the NY City and State CEI as previously described but also stock market indices.  
