# Graph-recommendation
Restaurant recommendation system based on a bipartite graph (link prediction).

## What is this about?
This project is a restaurant recommendation system based on a bipartite graph that uses information about previous restaurant visits by a set of people, alongside their reviews. It was originally developed using a NYC restaurant check-ins dataset from Forsquare (https://www.kaggle.com/datasets/danofer/foursquare-nyc-rest).

The approach is as follows: build a bipartite graph matching users to venues they have visited and enjoyed (based on the reviews), use the graph and visit information to build a feature vector for each user-restaurant match (restaurant sentiment, average number of visit to venue, graph similarity metrcs like Jaccard index, etc.), and then use the vectors to train a binary classificatiob model to predict link (1) vs no link (0).

## What is included?
The repository includes all necessary source code (data preprocessing and graph construction, sentiment and similarity calculations, model training, feature importance analysis), as well as a report describing the approach and results in detail.


<sup><sub>Note: the code is adapted by Olesia Khrapunova from a project from ML in Network Science course (MS in Data Science and Business Analytics, CentraleSupélec). The original code was written by Valentina Jerusalmi, Lucas de Souza Balancin, Rodolphe Royer de la Bastie and Olesia Khrapunova.</sub></sup>

