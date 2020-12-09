# E-Commerce-Company_Project

Let's suppose we just got some contract work with an Ecommerce company based in New York City that sells clothing online but they also have in-store style and clothing advice sessions. Customers come in to the store, have sessions/meetings with a personal stylist, then they can go home and order either on a mobile app or website for the clothes they want. Now, as a Data Scientist, we have to perform following four tasks for this company in order to maintain their overall revenue and marketing/advertisement cost to get the best results.

1. First Task:- The company is trying to decide whether to focus their efforts on their mobile app experience or their website. To solve this issue,we will keep record of average time spent on app in minutes, average time spent on website in minutes, length of membership of the peculiar customer in order to determine Yearly Amount Spent by the company by deploying various machine learning models. We will first to do exploratory data analysis in order to get an idea about the data we have, to understand the relationship between different features and also based upon the plots we will try to anticipate what can be the final answer for the problem so that even the company can understand it better. After that, we will do data pre-processing which includes data cleaning,splitting of data, feature scaling and encoding the categorical data. After that we will start with the deployement of machine learning algorithms. For this part, since our data has only 3 features and that too are important for getting the dependent variable, so feature selection is not necessary for this task.
We will use all types of regression techniques which include Multiple Linear Regression, Polynomial Regression, Decision Tree Regression, Random Forest Regression, Support Vector Regression and a Deep Learning Neural Network with two hidden layers. To improve the model performance, we will also apply model selection techniques like K-Fold Cross Validation and Hyperparameter Tuning using Randomized Search and Grid Search. In the end, we will also compare these models by checking the error functions like MAE, MSE, RMSE and R2_Score(since the data is linear) to get the best model. This dataset contains 500 customers who use these online platforms in regular intervals of time. 

Features Used:-
Avg. Session Length: Average session of in-store style advice sessions.
Time on App: Average time spent on App in minutes
Time on Website: Average time spent on Website in minutes
Length of Membership: How many years the customer has been a member.

Yearly Amount Spent:- Dependent Variable

Results:-
1. Multiple Linear Regression



Also, we will be working with a fake advertising data set, indicating whether or not a particular internet user clicked on an Advertisement on a company website. We will try to create a model that will predict whether or not they will click on an ad based off the features of that user.  This data set contains the following features:  

We'll work with the Ecommerce Customers csv file from the company. It has Customer info, suchas Email, Address, and their color Avatar. Then it also has numerical value columns:

Age: cutomer age in years 
Area Income: Avg. Income of geographical area of consumer 
Daily Internet Usage: Avg. minutes a day consumer is on the internet 
Ad Topic Line: Headline of the advertisement 
City: City of consumer 
Male: Whether or not consumer was male 
Country: Country of consumer 
Timestamp: Time at which consumer clicked on Ad or closed window 
Clicked on Ad: 0 or 1 indicated clicking on Ad
