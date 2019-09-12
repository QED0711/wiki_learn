# Wiki Learn

### A curriculum development tool for the modern web

Find the deployed web app [here](https://wikilearn.netlify.com)
___

## Table of Contents:

#### 1. [Problem Statement](#problem-statement)
#### 2. [Project Outline](#project-outline)
#### 3. [Data](#data)
#### 4. [Recommendation System](#recommendation-system)
#### 5. [Classification Model](#classification-model)
#### 6. [Custom Classes](#custom-classes)
#### 7. [References](#references)
#### 8. [Linked Repositories](#linked-repositories)

___

## Problem Statement

If you want to learn something, chances are that the knowledge exists somewhere online. However, the quantity of information can seem nearly infinite sometimes, and can be difficult to navigate. Where should you begin? What knowledge is prerequisite to my topic of interest? How do I place this knowledge in a larger context?

***Wiki Learn*** attempts to address these issues by providing a simple and intuitive way to quickly sort and navigate the vast collection of knowledge available online. 

### Who is this for?

This is primarily intended for the individual who wants to pursue a deeper understanding of a topic, but is unsure of how to go about parsing the enormous database of information that exists online. 

As a secondary target audience, Wiki Learn can also be useful for teachers and instructors. Principally, It may be used to help instructors fine tune existing curricula by recommending additional topics and resources related to the main discipline.  

___

## Project Outline

There are two main parts to this project:

### 1. Recommendation System

The main element of this project is a graph based recommendation system based on data obtained from Wikipedia articles. The recommendation system identifies what other wikipedia articles are important and highly similar to a primary topic.

### 2. Classification Model

With the results from the recommendation system, I use a machine learning classification model to determine what articles constitute prerequisite knowledge, and what articles should be read after a baseline understanding is achieved to further master the topic. 

___

## Data

All the data for this project is pulled directly from the Wikipedia API in real time. This method of collecting data as it is requested has some distinct advantages:

- It is always up to date with the most recent articles posted on Wikipedia
- Costly resources for storing large amounts of data are unnecessary
- There is no limitation on articles or topics that are part of the data set, as it includes all of Wikipedia and not just some subset that I was able to download

The main disadvantage of this approach is that calculating new recommendations can take some time (usually between 1-5 minutes). However, I believe that the advantages gained more than compensate for this slight performance drawback. 

___

## Recommendation System

The recommendation system used here is graph based, and uses item-to-item collaborative filtering to generate recommendations. 

The process is as follows:

1. A user specifies a Wikipedia article that they are interested in.

2. This article becomes the central node in a network of articles. I find all the links to and from this central node, and then all the links to and from each of those nodes. After expanding this network to two levels, I have a graph structure where each node is a Wikipedia article, and each edge (connection) is a link from one article to another. This is a directed graph because I keep track of links to and from each node in the network. 

3.  Having created this graph structure, I then collection information about each node's importance to the graph as a whole, and its relation to the main user defined article.

#### Network Importance:

- **edges:** a nodes total number of connections (in or out)
- **in_edges:** the number of articles referencing a specific node
- **out_edges:** the number of articles a node links to 
- **centrality:** a node's eigenvector centrality score
- **page_rank:** the page rank score obtained from NetworkX (based on the Google page rank algorithm, measures how important an article is to the network based on how frequently it is linked to).
- **adjusted_reciprocity:** reciprocity is the ratio of how many out_edges are returned over the total number of out edges. Typical reciprocity scores do not distinguish between nodes with only a few edges vs nodes with several edges. Adjusted reciprocity scales the score by the total number of out_edges. 

#### Importance to Central Node:
- **category_matches_with_source:** the total number of categories shared between a node and the central node. 
- **shared_neighbors_with)entry_score:** A ratio of the shared neighbors between a node and the central node over the total number of neighbors of both of them. 
- **shortest_path_length_from_entry:** the shortest, unweighted path in the directed graph from the central node to the target node. 
- **shortest_path_length_to_entry:** the shortest, unweighted path in the directed graph from the target node to the central node. 
- **jaccard_similarity:** the Jaccard similarity score between a node and the central node (total number of articles that link to both nodes over the total number of articles that link to either node).
- **primary_link:** a boolean value indicating if a node is a primary link. A node is considered to be a primary link if its reference in the central article occurs in the introduction or the 'See Also' section. _Note: when validating results (see below) the 'See Also' links are not recorded as primary links so as not to influence the validation score._


4. With features collected for each node, I rank their similarity to the primary user defined node by splitting the features into two categories: Bonuses and Penalties. Bonuses are features that, if a higher value, indicate that a particular node is important to the main topic or the network as a whole. Penalties are the opposite. If a feature is in the penalty category and has a high value, it means that the node is less related to the main topic. My similarity rank is simply the sum of the Bonus terms divided by the sum of the penalty terms. 

#### Bonuses:
- shared categories with entry
- shared neighbors with entry
- centrality
- page rank
- adjusted reciprocity
-Jaccard similarity

### Penalties:
- shortest path length to/from central node
- degrees / mean degrees in the network

5. Each node will receive a similarity_score, and recommendations are based on those nodes which have the highest scores. 

### Recommendation Validation

Evaluating recommender systems can be a challenging task, and often requires real world trials or a large amount of labeled data. For this reason, I choose not to perform _evaluation_ test, but _validation_ tests. 

The logic behind a validation test for a recommendation system is as follows: 

> If I know some items that are highly related to another item, and my recommendations place those items high on the list, I can be reasonably confident that my results are valid. 

While a large amount of labeled data is not easily retrievable, many Wikipedia articles do have human labeled articles that are known to be highly related to the current topic. These are found in the 'See Also' section. 

![](https://github.com/QED0711/wiki_learn/blob/master/visuals/see_also.png?raw=true)

Using these 'See Also' links as a sudo ground truth, I validate my recommendations by seeing how highly rated the 'See Also' links are.  

As a visual display, here we can see how my custom similarity_rank performs compared to other collected features. Here, I am using the topic, 'Random Forest', as the central node in my network.

![](https://github.com/QED0711/wiki_learn/blob/master/visuals/similarity_rank_chart.png?raw=true)

We can read each column in the above chart as the top percentile containing all 'See Also' links. As we can see, my similarity_rank places all the 'See Also' links in the top 99.37th percentile. 

We can look at this validation test in another way. We can see what percentage of our recommendations we must traverse before we capture all the 'See Also' links.

![](https://github.com/QED0711/wiki_learn/blob/master/visuals/similarity_rank_chart_v2.png?raw=true)

Similar to the previous chart, we can see how my similarity_rank compares to several individual features. Each bar can be read as how many of the 'See Also' links were captured in a given percentage of our recommendations. For example, the dark blue bar in each column represents the percentage of 'See Also' links captured in the top 1% of recommendations. 

Given these results, it is reasonable to assume that the recommendations generated are valid. 

___

## Classification Model

Recommendations are only one part of this curriculum development tool. When a user wishes to know more about a particular topic, they will likely want to know if any of the generated recommendations are prerequisite knowledge, and which recommendations should be studied after a baseline understanding is achieved. 

To accommodate this, I developed machine learning models to determine if a recommendation should come before or after a given topic. 

To train these models, my colleagues and I hand labeled around 400 pieces of data. The data used here came from the numerical features extracted from my generated graph networks. In passing these features and labels into a machine learning model, the goal is to find some underlying relationship between a node's features and is relative position (before or after) to the central node. 

### Best Performing Model

After trying several different models, my best performing model was a Random Forest, achieving around 0.8 train accuracy and 0.7 test accuracy scores. 

The confusion matrix and AUC/ROC curve below shows how this random forest model performed on all of the hand labeled data.

![](https://github.com/QED0711/wiki_learn/blob/master/visuals/confusion_matrix_rf.png?raw=true)

![](https://github.com/QED0711/wiki_learn/blob/master/visuals/auc_roc_curve.png?raw=true)

___

## Custom Classes

To streamline the entire process described above, I built two custom classes to handle graph creation, recommendation, and classification. 

The entire recommendation and classification pipeline can be written in just a few lines of code:

```
# initialize a graph with a user specified central node (link to Wikipedia article)
gc = GraphCreator("https://en.wikipedia.org/wiki/Decision_tree")

# initialize a recommendation pipeline and pass in our gc object
rec = Recommender(gc)

# fit the recommender, and pass in a scaler to scale the final results
rec.fit(scaler=Normalizer)

# provide a trained classifier model and make label predictions
rec.predict(rf_classifier)

# return formatted results with labels
rec.format_results()
```
A more in-depth demonstration of this code with comments can be found in [`notebooks/RecommenderPipeline.ipynb`](https://github.com/QED0711/wiki_learn/blob/master/notebooks/RecommenderPipeline.ipynb)

___


## References

[A graph-based recommender system for a digital library](https://www.researchgate.net/publication/220923824_A_graph-based_recommender_system_for_digital_library)

> Discusses the use and implementation of graph based recommender systems, and various ways go about finding similarities between nodes in a network

[Citolytics - A Link-based Recommender System for Wikipedia](https://www.gipp.com/wp-content/papercite-data/pdf/schwarzer2017.pdf)

> A recommendation system built specifically for Wikipedia articles. Discusses the use of the 'See Also' links as a _Gold Standard_ for validation.

___

## Linked Repositories

This project has been deployed as a full stack [web app](https://wikilearn.netlify.com). The repositories for the back and front ends can be found at the following links:

#### [Flask Back End](https://github.com/QED0711/wiki_learn_flask) 

#### [React Front End](https://github.com/QED0711/wiki_learn_react) 

