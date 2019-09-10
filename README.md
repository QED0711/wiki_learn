# Wiki Learn

### A curriculum development tool for the modern web
___

## Table of Contents:

#### 1. [Problem Statement](#Problem_Statement)
#### 2. [Project Outline](#)
#### 3. [Data](#)
#### 4. [Recommendation System](#)
#### 5. [Classification Model](#)
#### 6. [References](#)
#### 7. [Linked Repositories](#)

___

## Problem Statement

If you want to learn something, chances are that the knowledge exists somewhere online. However, the quantity of information can seem nearly infinite sometimes, and can be difficult to navigate. Where should you begin? What knowledge is prerequisite to my topic of interest? How do I place this knowledge in a larger context?

***Wiki Learn*** attempts address these issues by providing a simple and intutive way to quickly sort and navigate the vast collection of knowledges available online. 

#### Who is this for?

This is primarily intended for the individual who wants to pursue deeper knowledge of a topic, but is unsure of how to go about parsing the enormous database of knowledge that exists online. 

As a secondary target audience, Wiki Learn can also be useful for teachers and instructors. Principally, It may be used to help instructors fine tune existing curricula by recommending additional topics and resources related to the main discipline.  

___

## Project Outline

There are two main parts to this project:

### 1. Recommendation System

The main element of this project is a graph based recommendation system based on data obtained from Wikipedia articles. The recommendation systeme identifies what other wikipedia articles are important and highly similar to a primary topic.

### 2. Classification Model

With the results from the recommendation system, I use a machine learning classification model to determine what articles constitute prerequisite knowledge, and what articles should be read after a baseline understanding is achieved to further master the topic. 

___

## Data

All the data for this project is pulled directly from the Wikipedia API in real time. This method of collecting data as it is requested has some distinct advantages:

- It is always up to date with the most recent articles posted on Wikipedia
- There is no requirement for a large data store
- There is no limitation on articles or topics that are part of the dataset, as it includes all of Wikipedia and not jsut some subset that I was able to download

The main disadvantage of this approach is that calculating new recommendations can take some time (usually between 1-5 minutes). However, I believe that the advantages gained more than compensate for this slight performance drawback. 

___

## Recommendation System

The recommendation system used here is graph based, and uses item-to-item collaborative filtering to generate recommendations. 

The process is as follows:

1. A user specifies a Wikipedia article that they are interested in.

2. This article becomes the central node in a network of articles. I find all the links to and from this central node, and then all the links two and from each of those nodes. After expanding this network to two levels, I have a graph structure where each node is a Wikipedia article, and each edge (connection) is a link from one article to another. This graph structure is a directed graph because I keep track of links to and from each node in the network. 

3.  Having created this graph structure, I then collectin information about each node's importance to the graph as a whole, and its relation to the main user defined article. 

4. With features collected for each node, I rank their similarity to the primary user defined node by splitting the features into two categories: Bonuses and Penalties. Bonuses are features that, if a higher value, indicate that a particular node is important to the main topic of the network as a whole. Penalties are the opposite. If a feature is in the penalty category and has a high value, it means that the node is less related to the main topic. My similarity rank is simply the sum of the Bonus terms devided by the sum of the penalty terms. 

5. Each node will receive a similarity_score, and recommendations are based on those nodes which have the highest scores. 

### Recommendation Validation

Evaluating recommender systems can be a challenging task, and often requires real world trials or a large amount of labeled data. For this reason, I choose not to perform _evaluation_ test, but _validation_ tests. 

The logic behind a validation test for a recommendation system is as follows: 

> If I know some items that are highly related to another item, and my recommendations place those items high on the list, I can be reasonably confident that my results are valid. 





