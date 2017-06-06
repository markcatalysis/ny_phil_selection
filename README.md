# ny_phil_selection


# Motivation

The goal of this project is to track with time what compositions and composers are chosen for the NY Philharmonic's programs as a function of economic indicators at the local, state, and national levels. The reasoning is as follows. Conductors choose programs that balance personal preference and budgetary constraints. Conventional programs that largely feature crowd favorites like Beethoven and Mozart are more likely to fill seats than more modern or lesser known composers such as Mahler and Shostakovich who  often require larger orchestras for their pieces. During periods of economic downturn, attendance will drop. In order to compensate, program conventionality increases. By tracking economic indicators, one may be able to predict how conventional or unconventional the choice of music will be at the symphony.     

# Data Sources:

I am using publicly available data from https://github.com/nyphilarchive/PerformanceHistory. The sources of the economic indicators are the National Endowment of the Arts  https://www.arts.gov/news/2017/latest-economic-data-tracks-arts-and-cultural-jobs-state, the Federal Reserve https://fred.stlouisfed.org/categories, and the Federal Reserve Bank of New York   https://www.newyorkfed.org/research/regional_economy/coincident_summary.html.

# Methodology:

I am focusing primarily on degrees of unconventionality for the program data. In order to measure unconventionality vs economic health I first created a metric to label an individual work's degree of conventionality. Over the whole corpus of the NY Philharmonic dating back to 1842, I counted the total frequency of each unique composer and piece and for each work I gave it an unconventinality of (1/composer frequency)x(1/work frequency). This value yields a maximum of 1 as for a uniquely chosen work. It also "punishes" pieces such as Handel's Messiah which only appears 12 times but occurs regularly and seasonally. For an individual program I took the arithmetic mean of conventionality across the program. For an entire season, I took the arithmetic mean of unconventionality across all programs.

The economics data spanned only from the early 1970s on with many of the indicators spanning over much shorter periods. To account for these gaps, I have used forward fill to presume no change since the last indicator measure and subsequent back fill for all the rest. The daily economics data was matched to the first concert date for each program.

Included in my coding for the economics data are functions to shift the data back in time to represent response time lag and a delta function that create new columns that are the change in an economic indicator over a set time. Treating these as hyper parameters we can see if there is a stronger relationship between economic upturns with unconventionality and how long the choice of content for seasonal programming takes to respond.

From sklearn I used the RandomForestClassifier, LogisticRegression, and LinearRegression tools to perform my analyses. Because these are time dependent observations, I used TimeSeriesSplit for my test validation and removed time specific data such as date from the training. For the first two methods, I used the median program unconventionality of the whole NYPhil corpus as the threshold. In order to minimize splits over potentially-redundant features, I performed an unlabeled PCA transformation on the economics data before entering it into the random forest algorithm.  

As will be discussed later, the unconventionality by season is a more relevant metric as the choice of an individual program is made simultaneously with the other programs for a given season.

# Results:

Borrowing images from my capstone presentation for this project, we can see unconventionality varies greatly across  both programs and seasons. From the figure below, in both cases, the median is significantly smaller than the mean implying that the majority of performances were highly conventional largely composed of the same pieces by the same composers. This separation of programs speaks to the benefit of choosing this particular metric for unconventionality which allows for easy detection of less frequently performed works.

![Unconventionality Definition and Basic Statistics](https://github.com/markcatalysis/ny_phil_selection/blob/master/presentation/2017-06-06%20(2).png?raw=true)
Figure 1. Unconventionality across programs and seasons at a glance. Seasons largely smooth out unconventionality as would be expected from averaging across multiple programs but still the unconventionality value still captures similar choice behavior by conductors - highly conventional pieces that meet crowd expectations across most programs being the norm.   

By taking a look across across some of the chosen indicators and comparing them to unconventionality over time we can see that these fluctuations do not directly correspond to the degree of unconventionality seen over time, but there may be some correlation and a time delay that may suggest a delayed response in unconventionality to changes in economic health. Instead of a direct linear regression method, we will instead seek to detect high conventionality as defined by a program or season being above the median for all programs and seasons, respectively.  

![An Example Indicator, the NASDAQ, and Season Unconventionality with Time](https://github.com/markcatalysis/ny_phil_selection/blob/master/presentation/2017-06-06%20(5).png?raw=true)
Figure 2. NASDAQ and unconventionality by season over time.  

If we are to run our models and include both time delay and deltas in indicators over time, we find improved performance across various score metrics compared to the base model by adding time dependence. In every model I ran with added features corresponding to responses in time spanning 28 days to 6 months, I found increased predictive power in the models even with dropping the initial featurized data to mitigate naively overfitting to noise captured in added features. In particular, the best performing model for both logistic regression and random forest classification used were those with the added deltas over 28 days.

![Model Performance](https://github.com/markcatalysis/ny_phil_selection/blob/master/presentation/2017-06-06%20(6).png?raw=true)
Figure 3. Table of scores across various models.

One should note that predicting unconventionality over seasons largely improved predictive power. This makes intuitive sense in that programs are largely decided at the beginning of a season, so there is some room for variation within a season but the amount of conventionality for a whole season is set at the beginning. Additionally, the timing of the programming decisions necessitates that when a given program is performed is largely decoupled from the economic indicators at the time of performance but rather are better associated with those at the beginning of the season and thus are poorly described by the indicator values of the particular concert dates of that program.
