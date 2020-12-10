# E-Commerce-Company_Project

Let's suppose we just got some contract work with an E-commerce company based in New York City that sells clothing online. In addition to this, they also have in-store style and clothing advice sessions. Customers come into the store, have sessions/meetings with a personal stylist, then they can go home and order either on a mobile app or website for the clothes they want. Now, as a Data Scientist, we have to perform the following four tasks for this company to maintain their overall revenue and marketing/advertisement cost to get the best results.


1. First Task:- The company is trying to decide whether to focus its efforts on its mobile app experience or its website. To solve this issue, we will keep a record of average time spent on the app in minutes, average time spent on the website in minutes, length of membership of the peculiar customer to determine the Yearly Amount Spent by the customer by deploying various machine learning models. 

We will first do exploratory data analysis to get an idea about the data we have, to understand the relationship between different features, and also based upon the plots we will try to anticipate what can be the final answer for the problem so that even the board members in the organization can understand it better. After that, we will do data pre-processing that includes data cleaning, splitting of data, feature scaling, and encoding the categorical data. After that, we will start with the deployment of machine learning algorithms. For this part, since our data has only four features and that too is important for getting the dependent variable, so feature selection is not necessary for this task.

We will use eight types of regression techniques which include Multiple Linear Regression, Polynomial Regression, Decision Tree Regression, Random Forest Regression, Support Vector Regression, XGBoost Regression, Catboost Regression, and a Deep Learning Neural Network with two hidden layers. To improve the model performance, we will also apply model selection techniques like K-Fold Cross-Validation and Hyperparameter Tuning using Randomized Search and Grid Search. In the end, we will also compare these models by checking the error functions like MAE, MSE, RMSE, and R2_Score(since the data is linear) to get the best model. This dataset contains details of 500 customers who use these online platforms at regular intervals of time. 

Features Used:-
Avg. Session Length: Average session of in-store style advice sessions.
Time on App: Average time spent on App in minutes
Time on Website: Average time spent on Website in minutes
Length of Membership: How many years the customer has been a member.

Yearly Amount Spent by the Customer:- Dependent Variable

Results:-

1. Multiple Linear Regression:-
MAE: 7.645674798915295
MSE: 92.89010304498548
RMSE: 9.637951185028149
R2_Score: 98.28 %
Standard Deviation: 0.23 %

2. Polynomial Regression:-
MAE: 9.821412017434954
MSE: 168.1146222773757
RMSE: 12.965902293221852
R2_Score: 98.28 %
Standard Deviation: 0.23 %

3. Decision Tree Regression:-
MAE: 22.302801428455044
MSE: 892.9177245951145
RMSE: 29.881728942534675
R2_Score: 85.47 %
Standard Deviation: 4.47 %

4. Random Forest Regression:-
MAE: 17.312416565112212
MSE: 645.8748273882037
RMSE: 25.41406750971209
R2_Score: 93.31 %
Standard Deviation: 1.51 %

5. Support Vector Regression:-
MAE: 7.684301593806611
MSE: 93.41165186472095
RMSE: 9.664970349914217
R2_Score: 98.28 %
Standard Deviation: 0.23 %

6. XGBoost Regression:-
MAE: 12.416217399641823
MSE: 300.99728221487806
RMSE: 17.349273247455585
R2_Score: 96.55 %
Standard Deviation: 0.98 %

7. CatBoost Regression:-
MAE: 11.80193070072924
MSE: 290.54804106706683
RMSE: 17.04546981068773
R2_Score: 96.29 %
Standard Deviation: 1.07 %

8. Deep-Learning Neural Network:-
MAE: 8.977470664848756
MSE: 145.9051320075004
RMSE: 12.079119670220194
R2_Score: 97.25%
Standard Deviation: 0.994%


So, as we can see from the following results Multiple Linear Regression and Support Vector Regression have the best performance if we look at the final values, and it is because since the data is somehow linearly separable as in this case relationship between the yearly amount spent by the customer and length of membership of a particular customer is linearly separable as well as the relationship between the yearly amount spent by the customer and time spent on the app by the customer has a linear relationship.
And hence this is the final answer we got:-
Holding all other features fixed, a 1 unit increase in Avg. Session Length is associated with an increase of 25.98 total dollars spent by the customer.
Holding all other features fixed, a 1 unit increase in Time on App is associated with an increase of 38.59 total dollars spent by the customer.
Holding all other features fixed, a 1 unit increase in Time on Website is associated with an increase of 0.19 total dollars spent by the customer.
Holding all other features fixed, a 1 unit increase in Length of Membership is associated with an increase of 61.27 total dollars spent by the customer.

For this particular question:- Do you think the company should focus more on its mobile app or its website?
This is tricky, there are two ways to think about this: Develop the Website to catch up to the performance of the mobile app, or develop the app more since that is what is working better. This sort of answer depends on the other factors going on at the company, you would probably want to explore the relationship between Length of Membership and the App or the Website before concluding!


2. Second Task:- We will be working with an advertising data set, indicating whether or not a particular internet user clicked on an Advertisement on a company website or the app. These advertisements may be related to some other company's products and may be this companny has some tie-up with other companies in order to increase profits. We will try to create a model that will predict whether or not they will click on an ad based off the features of that user.  This data set contains the following features:  We'll work with the Ecommerce Customers csv file from the company. It has Customer info, suchas Email, Address, and their color Avatar. Then it also has numerical value columns.

Features Used:-
Age: cutomer age in years 
Area Income: Avg. Income of geographical area of consumer 
Daily Internet Usage: Avg. minutes a day consumer is on the internet 
Ad Topic Line: Headline of the advertisement 
City: City of consumer 
Male: Whether or not consumer was male 
Country: Country of consumer 
Timestamp: Time at which consumer clicked on Ad or closed window 
Clicked on Ad: 0 or 1 indicated clicking on Ad

We will first do exploratory data analysis to get an idea about the data we have, to understand the relationship between different features, and also based upon the plots we will try to anticipate what can be the final answer for the problem so that even the board members in the organization can understand it better. After that, we will do data pre-processing that includes data cleaning, splitting of data, feature scaling, and encoding the categorical data. After that, we will start with the deployment of machine learning algorithms. For this part, since our data has only five features and that too is important for getting the dependent variable, so feature selection is not necessary for this task.

We will use eight types of classification techniques which include Logistic Regression, KNearestNeighbour Classification, Decision Tree Classification, Random Forest Classification, Support Vector Machine Classification, XGBoost Classification, Catboost Classification, and a Deep Learning Neural Network with two hidden layers. To improve the model performance, we will also apply model selection techniques like K-Fold Cross-Validation and Hyperparameter Tuning using Randomized Search and Grid Search. In the end, we will also compare these models by comparing their accuracies to get the best model. This dataset contains details of 1000 internet users who use these online platforms at regular intervals of time. 

Results Obtained:-

1. Logistic Regression:-
Accuracy: 96.53 %
Standard Deviation: 1.48 %

2. KNearestNeighbour Classification:-
Accuracy: 96.27 %
Standard Deviation: 1.55 %

3. SVM Classification:-
Accuracy: 96.80 %
Standard Deviation: 1.36 %

4. Naive Bayes Classification:-
Accuracy: 96.27 %
Standard Deviation: 1.31 %

5. Decision Trees Classification:-
Accuracy: 95.07 %
Standard Deviation: 2.74 %

6. Random Forest Classification:-
Accuracy: 95.73 %
Standard Deviation: 1.87 %

7. XGBoost Classification:-
Accuracy: 95.60 %
Standard Deviation: 1.58 %

8. CatBoost Classification:-
Accuracy: 95.87 %
Standard Deviation: 1.73 %

9. Deep Learning Neural Network:-
Accuracy: 96.1333%
Standard Deviation: 2.4184476%

So, as we can see Support Vector Machine has the highest accuracy i.e. 96.80 which is using RBF as a kernel.
