# Bait 'n' Switch
## This One Simple Model has EVERY Marketing Firm Furious

![img](./src/images/clickbaitImage.jpg)

### Overview

For this project, I assumed the mindset of a data scientist working for a small, yet competitive internet browser startup. The stakeholders are looking for an edge on the larger browsers by implementing some quality of life improvements. The executives have asked me to create an extension for the browser that will filter out `click-bait` from webpages.

The first step in the process was building the classifier. Using natural language processing on manually labelled datasets <sup>[1](#myfootnote1)</sup><sup>[2](#dataset2)</sup><sup>[3](#dataset3)</sup> , I was able to produce a statistical model that could predict click bait with close to 90% accuracy (0.88 f1 score).

The next step in this project is to build out a browser extension. I plan on using a combination of regular expressions and the python module `Beautiful Soup` to creat a function that takes in as it’s argument raw html and returns a webpage that has the click-bait filtered from it. 


### Context
> "While we can't blame all attention distraction on click bait,
> as with any addictive relationship,
> it is an enabler."
- Gloria Mark is a professor specializing in human-computer interactions at the University of California, Irvine.

Clickbait can be defined as a headline or hyperlink that is intentionally designed to grab 
readers attention and are typically sensationalist or misleading. The superficial consequences of clickbait are the erosion of trust in the webpage or a perceived lack of credibility of a news source. For businesses built on an economy of clicks, where the more views a page gets, the more the business can charge their advertisers, click bait is a natural design evolution. However, now that this style of heading is pervasive in social media and in news agencies, the ability for malicious actors to add false information into headlines is becoming easier. Detecting fake news directly is difficult, but click bait sensationalism can act as a proxy for fake news articles . Therefore, I wanted to create a feed-back loop that penalizes news agencies for producing clickbait and frees up the user’s web browsing experience.

Reducing the amount of clickbait from the browsing experience has two major advantages for the browser of choice: 
The reduction in click bait will likely increase the usage of the browser due to the reduced cognitive load brought about by information overload.
Removing click bait from the browser will aid in the protection non-tech natives from accessing or spreading false information  

Internet privacy and data integrity are the foremost issues in big tech. By adding a feature that protects users from the spread of fake news while simultaneously boosting usership on the browsing platform is an absolute win. The benefit of providing a negative feed-back loop to news agencies who have built their empire around sensationalism is just gravy. 

### The Model

### The Data

#### Dataset 1
 - From Chakraborty et al., 2016 "Stop Clickbait: Detecting and preventing clickbaits in online news media" <sup>[1](#dataset1)</sup>
 - 32,000 news headlines, 16,000 clickbait and 16,000 non-clickbait articles
 -  The clickbait corpus consists of article headlines from ‘BuzzFeed’, ‘Upworthy’, ‘ViralNova’, ‘Thatscoop’, ‘Scoopwhoop’ and ‘ViralStories’. The non-clickbait article headlines are collected from ‘WikiNews’, ’New York Times’, ‘The Guardian’, and ‘The Hindu’.
 - Data was manually labelled, 3 labellers and the majority was taken as truth

#### Dataset 2
- The Webis Clickbait Corpus 2016 (Webis-Clickbait-16) comprises 2992 Twitter tweets sampled from top 20 news publishers as per retweets in 2014 <sup>[2](#dataset2)</sup>.
-  Data was manually labelled by 5 labellers on a scale of 1 to 5 and the mean was taken as truth.
- A total of 767 tweets are considered clickbait by the majority of annotators and 2225 where classified as normal
- The dataset contains the raw tweets containing urls. The urls are links to news headlines. I used those urls as tests for the model. 

#### Dataset 3

- The Webis Clickbait Corpus 2017 (Webis-Clickbait-17) comprises a total of 38,517 Twitter posts from 27 major US news publishers <sup>[3](#dataset3)</sup>. 
- All posts were annotated on a 4-point scale: not click baiting (0.0), slightly click baiting (0.33), considerably click baiting (0.66), heavily click baiting (1.0) by five annotators from Amazon Mechanical Turk
- 9,276 posts are considered clickbait by the majority of annotators and 29,241 where considered normal
![img](./src/images/classimbance.png)


The final
### Exploratory Data Analysis

I didn't realize how unconventional this project was going to be. It was my first NLP project and I had learnt somethings were important:
- Stem and lemmetise your corpus
- remove stopwords

It turned out that both of things damaged my models predictability. Two key findins where the presence of cardinal numbers and the proportion of stopwords present in either class

![img](./src/images/cardinality.png)

70% of all clickbait articles in my corpus contained cardinal numbers. This made a lot of sense to me. The number of "listicles" online are growing. In addition, there are the "17 surprising facts about bald eagles you should know" type headlines.

![img](./src/images/stopwords.png)
This was more surprising. Clickbait tends to have 20% more stopwords in each title than normal headlines. 

I believed I could leverage these class disparities using a Bag of Words approach. A Bag of Words is where you treat the words themselves as the features of the model. You then look at the corpus statistics. For instance, how many times does the word "frequency" appear in corpus. You can then take that and ask how many times frequency occurs in each class and you begin to get a sense of which words are more common to subclasses. For this study, I used Term Frequency-Inverse Document Frequency (Tf-idf) which is simply the number of times a word appears within a document weighted by the inverse of the number of times that word appears the corpus. 

The Disadvantage to bag of words is that during tokenization, you produce as many features as there are words. If you increase the n_gram range, which is the number of successive words that can be linked together into a token, you can create truly enormous matrices. 

My matrix that I used had over 50k rows and over 300k columns so it was very important that I use a statistical model that can evaluate quickly.

### Evaluation
![img](./src/images/baselinef1.png)

My baselines were all in the low to mid 80's for f1 score. Compared to the dummy classifier set to "most frequent" where it simply guesses the dominant class everytime.
### Discussion


### How to


### Repository Structure
```
.
├── bait'n'switch.ipynb
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
- <a name="dataset1">[1]</a>: Abhijnan Chakraborty, Bhargavi Paranjape, Sourya Kakarla, and Niloy Ganguly. "Stop Clickbait: Detecting and Preventing Clickbaits in Online News Media”. In Proceedings of the 2016 IEEE/ACM International Conference on Advances in Social Networks Analysis and Mining (ASONAM), San Fransisco, US, August 2016.
- <a name="dataset2">[2]</a>:  Potthast, Martin, Stein, Benno, Hagen, Matthias, & Köpsel, Sebastian. (2016). Webis Clickbait Corpus 2016 (Webis-Clickbait-16) [Data set]. Presented at the 38th European Conference on IR Research (ECIR 2016), Zenodo.
- <a name="dataset3">[3]</a>: Potthast, Martin, Gollub, Tim, Wiegmann, Matti, Stein, Benno, Hagen, Matthias, Komlossy, Kristof, … Fernandez, Erika P. Garces. (2018). Webis Clickbait Corpus 2017 (Webis-Clickbait-17) [Data set].
- <a name="dataset2">[4]</a>: Bird, Steven, Edward Loper and Ewan Klein (2009), Natural Language Processing with Python. O’Reilly Media Inc.
