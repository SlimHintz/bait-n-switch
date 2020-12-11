# Bait 'n' Switch
## A tool for decluttering a busy browsing experience

### Overview

For this project, I assumed the mindset of a data scientist working for a small, yet competitive internet browser startup. The stakeholders are looking for an edge on the larger browsers by implementing some quality of life improvements. The executives have asked me to create an extension for the browser that will filter out `click-bait` from webpages.

Using natural language processing on manually labelled datasets[1][2][3], I was able to produce a statistical model that could predict click bait with 98% accuracy. Then, using a combination of regular expressions and the python module `Beautiful Soup` I created a function that takes in as it’s argument raw html and returns a webpage that has the click bait filtered from it.


### Context
>> While we can't blame all attention distraction on click bait, as with any addictive relationship, it is an enabler.

### The Model

### The Data

#### Dataset 1
 - From Chakraborty et al., 2016 "Stop Clickbait: Detecting and preventing clickbaits in online news media" [1]
 - 32,000 news headlines, 16,000 clickbait and 16,000 non-clickbait articles
 -  The clickbait corpus consists of article headlines from ‘BuzzFeed’, ‘Upworthy’, ‘ViralNova’, ‘Thatscoop’, ‘Scoopwhoop’ and ‘ViralStories’. The non-clickbait article headlines are collected from ‘WikiNews’, ’New York Times’, ‘The Guardian’, and ‘The Hindu’.
 - Data was manually labelled, 3 labellers and the majority was taken as truth

#### Dataset 2
- The Webis Clickbait Corpus 2016 (Webis-Clickbait-16) comprises 2992 Twitter tweets sampled from top 20 news publishers as per retweets in 2014 [2]
-  Data was manually labelled, 3 labellers and the majority was taken as truth
- A total of 767 tweets are considered clickbait by the majority of annotators and 2225 where classified as normal

#### Dataset 3

- The Webis Clickbait Corpus 2017 (Webis-Clickbait-17) comprises a total of 38,517 Twitter posts from 27 major US news publishers.
- All posts were annotated on a 4-point scale: not click baiting (0.0), slightly click baiting (0.33), considerably click baiting (0.66), heavily click baiting (1.0) by five annotators from Amazon Mechanical Turk
- 9,276 posts are considered clickbait by the majority of annotators and 29,241 where considered normal

Data was then evaluated seperately, cleaned seperately and then disimilarity matrices were constructed to evaluate similarity between corpus. 

N closest headlines were added to the first data set from the other two based on similarity.

### Evaluation


### Discussion


### How to


### Repository Structure
```
.
├── Index.ipynb
├── README.md
├── notebooks
│   ├── EDA
│   └── modelling
└── src
    ├── data
    ├── models
    └── modules
```
### References
- [1] Abhijnan Chakraborty, Bhargavi Paranjape, Sourya Kakarla, and Niloy Ganguly. "Stop Clickbait: Detecting and Preventing Clickbaits in Online News Media”. In Proceedings of the 2016 IEEE/ACM International Conference on Advances in Social Networks Analysis and Mining (ASONAM), San Fransisco, US, August 2016.
- [2] Potthast, Martin, Stein, Benno, Hagen, Matthias, & Köpsel, Sebastian. (2016). Webis Clickbait Corpus 2016 (Webis-Clickbait-16) [Data set]. Presented at the 38th European Conference on IR Research (ECIR 2016), Zenodo.
- [3] Potthast, Martin, Gollub, Tim, Wiegmann, Matti, Stein, Benno, Hagen, Matthias, Komlossy, Kristof, … Fernandez, Erika P. Garces. (2018). Webis Clickbait Corpus 2017 (Webis-Clickbait-17) [Data set].
