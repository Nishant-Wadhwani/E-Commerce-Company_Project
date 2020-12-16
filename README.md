# E-Commerce-Company_Project
 
Let's suppose we just got some contract work with an E-commerce company named Tech_novan based in New York City, an online superstore for electronic gadgets and computer products. The website has a wide range of products ranging from laptops, tablets, Smartphones, Digital Cameras to Gaming hardware, monitors to printers, routers, and speakers. Shoppers can browse and buy a wide variety of electronics on the extensive but easy-to-navigate e-commerce site.

The navigation and categorization are well defined, giving you suggestions for keywords as you type. In addition to this, they also have in-store sessions related to electronic gadgets or device's functionality and provide feedback related to a particular model in these advice sessions. Customers come into the store, have sessions/meetings with a personal techie, then they can go home and order either on a mobile app or website for the device/gadget they want. Now, as a Data Scientist, we have to perform the following six tasks for this company to maintain their overall revenue and marketing/advertisement cost to get the best results.


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


2. Second Task:- In this peculiar task, We're going to optimize the online advertising by going to find the best ad among different ad designs, the best ad that will converge the maximum customers to click on the ad and potentially buy the company's product.
The advertising team prepared ten different ads with ten different designs. For example like in one of the ads the customers will see the smartphone commercial done by famous cricketer Virat Kohli, who is doing heavy work-out on the field and recording/monitoring his performance on some app or comparing his metabolism with battery-life, on the other ad customers see the smartwatch commercial done by renowned actor Brad Pitt during some action scene while chasing criminals, and on some other ad say some tourists models are using a digital camera which is taking HD photographs of an SUV in a charming city like the south of France or Italy, etc.  Now, the advertising team is wondering which ad will converge the most, which ad will attract the most people to click the ad and then potentially buy their products. 

So we have these ten different ads and what we're going to do is the process of online learning, which is to show these ads to different users online. Once they connect to a certain website or a search engine, it can be ads that appear at the top of a page when they do any type of research on Google. We will show one of these ads each time the user connects to the Web page and we're going to record the result whether this user clicked yes or no on the ad.
We will be using reinforcement learning for this task that includes Upper Confidence Bound and Thompson Sampling for getting the final result. So, we will select an ad to show to this user and then the user will decide to click yes or no on the ad. If the user clicks on the ad we will record it as one whereas if the user doesn't click on the ad we will record it as 0. And then a new user connects to the Web page and same the algorithm selects an ad to show this new user. Our dataset contains 10000 users. We need to figure out in a minimum number of rounds which ad converts to the most meaning which is the best ad to which the users are most attracted.

Results:-
Ad number five was the ad that is selected the most and was the ad with the highest click-through rate. In terms of our business case study, it corresponds indeed to the ad that is the most attractive ad that has the most fanbase associated, that will sell the most to the user as future customers.

The UCB algorithm did a great job but here is a constraint that is to identify this ad as soon as possible which is in a minimum number of rounds. We have to observe how many rounds are required for the UCB algorithm to able to identify this ad with the highest clicks. We will implement this by tweaking the number of customers as a parameter. With 5000 rounds that is with 5000 users, the UCB can identify the ad with the highest CTR. If we replace 5000 here with 1000 and still it can identify the ad with the highest CTR which is still ad 5. Now, we're going to replace that 1000 here with 500. And 500 rounds is not enough for the UCB algorithm to identify the best ad, the UCB identified the best ad as an ad of index 7 so 500 is not enough.

Now, if we use the Thompson Sampling algorithm with the same data set then it will give the correct result even with 500 users i.e. it was able to find the ad number 5 with the highest CTR even in 500 rounds. And therefore this technique is more powerful and efficient than UCB in most situations.


3. Third Task:- Now, as we got our best ad, that is ad no.5, we are going to test this ad for our new set of customers. We will be working with an advertising data set that contains 1000 new users and indicating whether or not a particular internet user clicked on an advertisement on any website or app. These advertisements will be related to our company's product that can be any electronic gadget or device because it's a high chance that these new customers will click on this ad since it has got maximum CTR when tested on our old customers. And if a person clicks on this best ad, it's a possibility that he/she will purchase our product. We will try to create a model that will predict whether or not they will click on an ad-based on the features of that user. We'll work with the Ecommerce Customer's CSV file from the company. It has Customer info, such as Email, Address, and their color Avatar. Then it also has numerical value columns. This data set contains the following features:  

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

We will use nine types of classification techniques which include Logistic Regression, KNearestNeighbour Classification, Decision Tree Classification, Random Forest Classification, Support Vector Machine Classification, XGBoost Classification, Catboost Classification, and a Deep Learning Neural Network with two hidden layers. To improve the model performance, we will also apply model selection techniques like K-Fold Cross-Validation and Hyperparameter Tuning using Randomized Search and Grid Search. In the end, we will also compare these models by comparing their accuracies to get the best model. This dataset contains details of 1000 internet users who use these online platforms at regular intervals of time. 

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

So, as we can see Support Vector Machine has correctly predicted the result with the highest accuracy i.e. 96.80 which is using RBF as a kernel which represents that the data here is non-linear in nature. Also, we got the magnificent result that is out of 520 women users, around 260 have clicked our ad to see the specific gadget and out of 480 men, around 230 have clicked our ad which shows in total around half of the new users found our ad engaging.

4. Fourth Task:- Now, after doing solving all these tasks, we are interested in the main problem that is whether the person is interested in buying the product or not i.e. whether the peculiar customer after attending all the advice sessions and clicked on the ad that has the highest CTR, has purchased any of the company's product. So, using classification, we are going to predict whether the customer has purchased the product or not based on two features that are the age of the customer and his estimated salary in dollars per annum. So, it is quite clear that from our previous dataset, around 500 customers have clicked the best design ad, so we are taking the information that is age and estimated salary of 400 customers.

We will first do exploratory data analysis to get an idea about the data we have, to understand the relationship between different features, and also based upon the plots we will try to anticipate what can be the final answer for the problem so that even the board members in the organization can understand it better. After that, we will do data pre-processing that includes data cleaning, splitting of data, feature scaling, and encoding the categorical data. After that, we will start with the deployment of machine learning algorithms. For this part, since our data has only two features and that too is important for getting the dependent variable, so feature selection is not necessary for this task.

We will use nine types of classification techniques which include Logistic Regression, KNearestNeighbour Classification, Decision Tree Classification, Random Forest Classification, Support Vector Machine Classification, XGBoost Classification, Catboost Classification, and a Deep Learning Neural Network with two hidden layers. To improve the model performance, we will also apply model selection techniques like K-Fold Cross-Validation and Hyperparameter Tuning using Randomized Search and Grid Search. In the end, we will also compare these models by comparing their accuracies to get the best model. This dataset contains details of 1000 internet users who use these online platforms at regular intervals of time. 


Results Obtained:-

1. Logistic Regression:-
Accuracy:  82.67 %
Standard Deviation: 9.52 %

2. KNearestNeighbour Classification:-
Accuracy: 91.00 %
Standard Deviation: 5.59 %

3. SVM Classification:-
Accuracy: 90.67 %
Standard Deviation: 6.11 %

4. Naive Bayes Classification:-
Accuracy: 87.67 %
Standard Deviation: 8.95 %

5. Decision Trees Classification:-
Accuracy: 90.67 %
Standard Deviation: 6.29 %

6. Random Forest Classification:-
Accuracy: Accuracy: 90.00 %
Standard Deviation: 7.30 %

7. XGBoost Classification:-
Accuracy: 90.33 %
Standard Deviation: 6.40 %

8. CatBoost Classification:-
Accuracy: 91.00 %
Standard Deviation: 5.97 %

9. Deep Learning Neural Network:-
Accuracy: 89.0
Standard Deviation: 5.972

So, as we can see KNN Classification has predicted with best average accuracy i.e. 91% whether the customer will buy the product or not after clicking the ad with the best design or the ad with maximum click-through rates. It means our model is 91 percent efficient in predicting the correct number of users who have either purchased the product or not. Also, out of the 400 customers, around 140 customers have brought something for them either from the company's website or app or through visiting the store which is tremendous for the company's profit.
