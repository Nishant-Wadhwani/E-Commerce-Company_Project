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


2. Second Task:- In this peculiar task, We're going to optimize the online advertising by going to find the best ad you know among different ad designs, the best ad that will converge the maximum customers to click on the ad and potentially buy the company's product.
The advertising team prepared ten different ads with ten different designs. For example like in one of the ads the customers will see the smartphone commercial done by famous cricketer Virat Kohli, who is doing heavy work-out on the field and recording/monitoring his performance on some app or comparing his metabolism with battery-life, on the other ad customers see the smartwatch commercial done by renowned actor Brad Pitt during some action scene while chasing criminals, and on some other ad say some tourists models are using a digital camera which is taking HD photographs of an SUV in a charming city like the south of France or Italy, etc.  Now, the advertising team is wondering which ad will converge the most, which ad will attract the most people to click the ad and then potentially buy their products. 

So we have these ten different ads and what we're going to do is the process of online learning, which is to show these ads to different users online. Once they connect to a certain website or a search engine, it can be ads that appear at the top of a page when they do any type of research on Google. We will show one of these ads each time the user connects to the Web page and we're going to record the result whether this user clicked yes or no on the ad.
We will be using reinforcement learning for this task that includes Upper Confidence Bound and Thompson Sampling for getting the final result. So, we will select an ad to show to this user and then the user will decide to click yes or no on the ad. If the user clicks on the ad we will record it as one whereas if the user doesn't click on the ad we will record it as 0. And then a new user connects to the Web page and same the algorithm selects an ad to show this new user. Our dataset contains 10000 users. We need to figure out in a minimum number of rounds which ad converts to the most meaning which is the best ad to which the users are most attracted.




Indeed this ad of index for here meaning the ad number five was clearly the ad selected the most therefore

was clearly the ad with the highest click through rate.

And in terms of our business case study it corresponds indeed to the ad that is the most attractive

meaning the ad that has the most beautiful image of that car that will sell the most to the user as

a future customers.

OK.

So good job.

The UCB algorithm did a very up here however remember that you know the girl is in fact to you know

identify this ad as fast as possible you know in a minimum number of rounds and therefore what we should

experiment right now.

You know it's not quite over.

We should experiment to see and actually how many rounds the UCB algorithm was able to identify this

ad with the highest city are and the way to check this is by you know changing that value.

And here because this algorithm was run with 10000 rounds.

But what if we put instead you know 5000 rounds you know we would like to see if the UCB is still able

to identify that ad of index for you know with the high city or in 5000 rounds.

So that's exactly what we're going to check now.

But you know rerun everything and the way to do this is by clicking run time here and then restart and

run.

And we're going to see if with 5000 rounds well at UCB algorithm is also able to quickly figure out

the best ad and yes very good.

Even with 5000 rounds you know with 5000 users the UCB was able to identify the ad with the highest

CAGR.

And now let's make it even more challenging to the UCB.

Let's replace 5000 here by 1000 and let's click runtime again and restart and run.

Oh and let's see if with only 1000 rounds while the UCB is able to.

Wow.

OK.

So still it was still able to do it you know identifying the ad with the high CGI which is still of

course ad of the next four but quite just you know.

So now I of course want to try with 500 rounds.

Right.

We're going to replace that 1000 here by 500 then rerun everything by clicking restart and run out.

And now let's see.

But I'm not sure it.

Let's see if it is still able to identify that ad with the high CGI.

And that's what I'm talking about you know 500 rounds is not enough for the UCB algorithm to identify

that best at you know that ad with the highest CAGR because indeed the ad with the high city or is clearly

at number four.

But in 500 round the UCB identified the best ad as ad of index 7 so 500 is not enough.

And so now it will be very interesting to see with the Thompson sampling algorithm if you know it can

beat the UCB algorithm in the sense that it can find this out of index for with the highest city are

in 500 rounds because they used to be can clearly find it in 1000 rounds.

That's for sure but not in 500 rounds.

So we're going to see with Thompson sampling algorithm if this one is able to find that best ad you

know because we're going to work on the same data set of course if it is going to identify the best

ad of index four in 500 rounds and if yes we will even try lower numbers of rounds.

So I can't wait to now implement Thompson something with you we will work on the exact same data set.
And unfortunately the UCB algorithm was not able to figure out the best ad you know that ad of index

4 with the highest conversion rate in 500 rounds right.

It identified ad number 7.

So what we really want to see now you know when visualizing the results of Thompson sampling is if Thompson

simply can not only figure out that same best ad you know with the highest conversion rate in 10000

rounds first we will start with 10000 rounds.

But mostly what we want to see is if it is able to figure out that best ad in less than 500 rounds or

you know 500 rounds because if that's the case then that means Thompson sampling will beat the UCB algorithm.

So can't wait to try.

I promise that I actually have no idea because you know this is a new recording that I'm doing and when

I made these implementations I only visualized the results with ten thousand so I'm discovering the

results with you.

And that's why I'm just super excited to not only show this to you but also to myself.

So we're all in the same row of seats you know watching the same show.

OK.

So let's do this enough talking.

Let's click this fully here to you know upload the data set into the notebook let's not forget to do

this because we actually have not run any cell so far.

So right now it is connecting to a runtime to enable file browsing and in a second we should see hopefully

the upload button.

It is not coming but you know sometimes it takes time.

There we go.

All right.

So the upload button.

Let's click it.

And now in your machine please find the machinery it is at folder.

Wherever you put it in which you had to download at the beginning of each section including this one

then once you find it let's go inside.

Let's go to part 6 now.

Reinforcement learning and Section 33 Thompson sampling and of course by phone and at CGI are optimization.

Let's click open.

Let's click Okay.

Now we have the data set.

So I suggest that we do a run out so that we can you know quickly experiment with the different numbers

of rounds and we're going to start with 10000 of course just to make sure that Thompson's sampling works

correctly.

So let's do this.

Let's click runtime here and then run all.

And now all the cells will be running including this one.

Everything seems to be all good and well well well that's actually even more incredible than the UCB

algorithm.

You know it seems that this ad of index 4 was actually very quickly identified as the best.

I'm actually very confident about that question whether or not the Thompson something will beat UCB

in the sense that it will be able to identify this ad of index for as the best one in less than 500

rounds.

Right.

Clearly all these other ads here are smashed.

With respect to this one.

Right.

If we have a look at UCB again see the other ads are way more selective here than you know this one.

OK.

So I'm really really confident about this.

So I think we can directly not write 500 but you know 1000.

So I'm going to remove a zero here.

All right.

And then we're just gonna do another runtime and then restart this time and run off so that we can restart

everything and rerun all the cells yes here and now all the cells will be running again with this 1000

rounds now and let's see the new results.

And coming in a second.

There we go.

OK.

So now with 1000 rounds of course the other ads are a bit higher.

You know I mean the bars of these other ads are a bit higher because of course now we're on a different

scale.

We're only with 1000 rounds but still with 1000 rounds.

Well that ad of index four was way more selective then the others and now time for the ultimate truth

will Thompson sampling beat the UCB algorithm.

Meaning.

Will it be able to identify the best ad with the highest conversion rate in 500 rounds.

Well that's what we're about to figure out right away.

So let's replace that 1000 value here by five hundred.

Then let's click runtime again and then restart and run out.

Are you ready.

We're about to reveal the ultimate truth.

Let's do this.

Yes.

And now.

Now all the sales will be running again.

And let's see if Thompson simply can figure out the best ad in 500 rounds.

And congratulations Thompson sampling.

It was indeed truly able to figure out that ad of index 4 with the highest conversion rate in as little

as 500 rounds.

So that was expected.

You know I actually expect this because I knew from my educational background in machine learning you

know reinforcement learning with one of the topics I had it's cool in my masters of machine learning

that indeed Thompson's sampling is more powerful than UCB in most situations and that's what we clearly

see here right.

Even in 500 rounds that add an index four was way more selected than the others you know almost twice

as much as this first ad and twice as much as this one.

So clearly Thompson something did an amazing job at quickly identifying that best ad with the highest

conversion rate.

So to the question Should I try.

Should I choose UCB or Thompson sampling.

Well of course you can try that too because you know it just takes a few seconds to run the two but

if you have a doubt.

Well I would suggest to go with Thompson sampling because India is more powerful you know it is faster

to identify the elements with the highest rates.

So there we go.

I was really happy to implement these two models with you UCB and Thompson something and I'm really

glad that we enjoyed at the same time the final results on our notebook and now we're gonna move on












We will be working with an advertising data set, indicating whether or not a particular internet user clicked on an Advertisement on a company website or the app. These advertisements may be related to some other company's products and may be this companny has some tie-up with other companies in order to increase profits. We will try to create a model that will predict whether or not they will click on an ad based off the features of that user.  This data set contains the following features:  We'll work with the Ecommerce Customers csv file from the company. It has Customer info, suchas Email, Address, and their color Avatar. Then it also has numerical value columns.

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
