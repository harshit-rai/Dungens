# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 17:11:02 2019

@author: avdes
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime


customer_data = pd.read_csv('olist_customers_dataset.csv')
general_geolocation_data = pd.read_csv('olist_geolocation_dataset.csv')
order_item_data = pd.read_csv('olist_order_items_dataset.csv')
order_payment_data = pd.read_csv('olist_order_payments_dataset.csv')
order_review_data = pd.read_csv('olist_order_reviews_dataset.csv')
orders_data = pd.read_csv('olist_orders_dataset.csv')
product_data = pd.read_csv('olist_products_dataset.csv')
sellers_data = pd.read_csv('olist_sellers_dataset.csv')
product_category_data = pd.read_csv('product_category_name_translation.csv')

for colname in prod_value_4:
    print(colname)
    
    
    
    
#_______________________________Customer_data_______________________________________________________________

#Statewise number of customer distribution
x = customer_data.drop_duplicates('customer_unique_id')
statewise_total_customers = pd.DataFrame(x.customer_state.value_counts())
statewise_total_customers
myplot = statewise_total_customers.plot.bar(figsize=(10,10))
myplot.set_ylabel('Number of unique customers') 
myplot.set_xlabel('States')
myplot.set_title('Number of unique customers per state')


#_______________________________Item_data______________________________________________________________

#Total frequency of each product sold and number of unique products
#Total products sold by each seller id and top 10 sellers, calculating freight cost %. 
#Mean freight value associated with each unique product.
#Mean product cost of unique products

product_frequency = pd.DataFrame(order_item_data.product_id.value_counts())
product_frequency.rename(columns={'product_id':'no_of_products_sold'})
product_frequency.rename(columns={'':'product_id'})
product_frequency         #Frequency of each unique product sold

no_unique_products = len(product_frequency)
no_unique_products          #Number of unique products
unique_products = order_item_data.product_id.unique()      #All unique product ids

items_sold_seller = pd.DataFrame(order_item_data.seller_id.value_counts())    #it shows total number of items sold by each individual
top_10_sellers = items_sold_seller.iloc[0:10]       #Top 10 sellers by number of items sold

order_item_data['Freight_cost_%'] = (order_item_data['freight_value'])*100/(order_item_data['total_cost'])
order_item_data['Freight_cost_%'].describe()                        #Freight cost analysis

unique_sellers = order_item_data.seller_id.unique()         #All unique sellers
mean_unique_seller = []

for j in range (len(unique_sellers)):
    temp = pd.DataFrame(order_item_data[order_item_data.seller_id == str(unique_sellers[j])])
    a = temp.freight_value.mean()
    mean_unique_seller.append(a)
    j +=1

df_uni_seller = pd.DataFrame({'unique_seller_id':unique_sellers,'mean_freight_cost':mean_unique_seller}, columns=['unique_seller_id','mean_freight_cost'])
df_uni_seller.sort_values('mean_freight_cost', ascending=False)           #Shows mean freight cost per seller 
df_uni_seller.describe()                    #Freight cost per seller analysis
df_uni_seller[df_uni_seller.mean_freight_cost >= 24.37].sort_values('mean_freight_cost', ascending=False)           #Sellers with mean cost above 3rd quartile

#product_fv = order_item_data[order_item_data.freight_value]

mean_unique_product=[]
mean_product_cost = []
product_freq = []


for k in range (no_unique_products):
    temp1 = pd.DataFrame(order_item_data[order_item_data.product_id == str(unique_products[k])])
    b = temp1.freight_value.mean()
    c = temp1.price.mean()
    d = temp1.product_id.count()
    mean_unique_product.append(b)
    mean_product_cost.append(c)
    product_freq.append(d)
    k +=1
                             
df_uni_product = pd.DataFrame({'product_id':unique_products,'mean_product_fv':mean_unique_product}, columns=['product_id','mean_product_fv'])       #df of unique products ids with their mean freight cost
df_uni_product_price = pd.DataFrame({'product_id':unique_products,'mean_product_cost':mean_product_cost}, columns=['product_id','mean_product_cost'])           #df of unique products ids with their mean product cost
df_product_freq = pd.DataFrame({'product_id':unique_products,'no_of_products_sold':product_freq}, columns=['product_id','no_of_products_sold'])         #df showing number of items sold for each unique product id

len(mean_unique_product )
len(mean_product_cost)
len(product_freq)

df_uni_product.describe()                    
product_high_fv = order_item_data[order_item_data.freight_value >= 24.37].sort_values('freight_value', ascending=False)         #Products with freight cost % greater than 23.7 in sorted order
product_high_fv.iloc[0:30,2:6]       #top 30 products with high freight values
t = pd.DataFrame(product_high_fv.product_id.unique())           
t.iloc[0:30]   #Procut ids of top 30 products with highest freight values.




                             
df_uni_product = pd.DataFrame({'product_id':unique_products,'mean_product_fv':mean_unique_product}, columns=['product_id','mean_product_fv'])

df_uni_seller = pd.DataFrame({'unique_seller_id':unique_sellers,'mean_freight_cost':mean_unique_seller}, columns=['unique_seller_id','mean_freight_cost'])






#___________________________________Payment_data___________________________________________________________

#Preference for payment modes,number of installments. 
payment_mode_freq = pd.DataFrame(order_payment_data.payment_type.value_counts())            #Calculates payment mode frequency
payment_modes = order_payment_data.payment_type.unique()            #All unique modes of payment
plt.pie(payment_mode_freq.iloc[:],labels=payment_modes, radius=3.5, autopct='%0.2f%%',explode=[0,0,0,0,0.5])        #pie chart of payment mode preffered
payment_installments = order_payment_data.payment_installments.unique()     #Number of unique installment numbers

payment_installment_freq = pd.DataFrame(order_payment_data.payment_installments.value_counts())    #frequency of each intallment number
plt.scatter(order_payment_data.iloc[:,3],order_payment_data.iloc[:,5])                      #scatter plot of payment installment frequency

#visualizing through other plots the payment installment frequency
sns.jointplot(x="payment_installments", y="total_cost", data=order_payment_data, kind="reg")
sns.jointplot(x="payment_installments", y="total_cost", data=order_payment_data, kind="hex")    


    
    
#______________________________________Review_data________________________________________________________

#Reviews frequency and average reviews
review_freq = pd.DataFrame(order_review_data.review_score.value_counts())   #frequency of unique reviews
mean_review = order_review_data.review_score.mean()             #Calculating average review
print('\nAverage review from customers is: ' + str(mean_review))

review_score = order_review_data.review_score.unique()
plt.pie(review_freq.iloc[:],labels=review_score, radius=3.5, autopct='%0.2f%%',explode=[0.05,0.05,0.05,0.05,0.05])          #pie chart to show review scores %

prod_value_4 = pd.merge(order_review_data, order_item_data, on='order_id')    #merged order_review_data and order_item_data dfs on basis of order id
review_mean = []

for m in range (no_unique_products):
    temp2 = pd.DataFrame(prod_value_4[prod_value_4.product_id == str(unique_products[m])])
    e = temp2.review_score.mean()
    review_mean.append(e)
    m +=1
len(review_mean)
df_prod_review = pd.DataFrame({'product_id':unique_products,'review_mean':review_mean}, columns=['product_id','review_mean'])    #shows review mean of each unique product





#______________________________________Order_data________________________________________________________

#Loss of revenue in terms of freight cost, instances where inventory fell short, delivery time difference from estimated delivery time 
#Check number of early, timely and late deliveries, calculating actual delivery time taken


approved_orders = orders_data[pd.notnull(orders_data['order_approved_at'])]
shipped_orders = approved_orders[pd.notnull(approved_orders['order_delivered_carrier_date'])]
never_delivered = shipped_orders[pd.isnull(shipped_orders['order_delivered_customer_date'])]
print('\n\nInstances where order was placed, approved, shipped but was never delivered to customer which resulted in loss/extra costing to company in terms of freight cost is: ' + str(len(never_delivered)))
orders_data.order_status.unique()
unavailable = len(orders_data[orders_data.order_status == 'unavailable'])
print('\n\nTotal number of instances where inventory fell short i.e product was not available is: ' + str(unavailable))


orders_data.loc[:,'order_delivered_customer_date'] = pd.to_datetime(orders_data.loc[:,'order_delivered_customer_date']) 
orders_data.loc[:,'order_estimated_delivery_date'] = pd.to_datetime(orders_data.loc[:,'order_estimated_delivery_date']) 
orders_data['difference'] = orders_data.loc[:,'order_estimated_delivery_date'] - orders_data.loc[:,'order_delivered_customer_date']

delivery_time_difference = orders_data[pd.notnull(orders_data['difference'])]
delivery_time_difference.describe()
print('\n\nAverage difference of days between estimated delivery date and actual delivery date is: ' + str(delivery_time_difference.difference.mean()) + ' hr:mn:sec')
Early = 0
On_time = 0
Late = 0

for i in range (len(delivery_time_difference['difference'])):
    if (delivery_time_difference.iloc[i,8] > pd.to_timedelta(0)):
        Early += 1
    if (delivery_time_difference.iloc[i,8] == pd.to_timedelta(0)):
        On_time += 1
    if (delivery_time_difference.iloc[i,8] < pd.to_timedelta(0)):
        Late += 1

pie_slice = [Early, On_time, Late]
pie_names = ['Early', 'On_time', 'Late']
plt.pie(pie_slice, labels=pie_names, radius=3.5, autopct='%0.1f%%',explode=[0.05,0.05,0.05] )
plt.legend(pie_names)



orders_data.loc[:,'order_purchase_timestamp'] = pd.to_datetime(orders_data.loc[:,'order_purchase_timestamp']) 
orders_data.loc[:,'order_delivered_customer_date'] = pd.to_datetime(orders_data.loc[:,'order_delivered_customer_date']) 
orders_data['actual_delivery_time'] = orders_data.loc[:,'order_delivered_customer_date'] - orders_data.loc[:,'order_purchase_timestamp']

actual_delivery_time = orders_data[pd.notnull(orders_data['actual_delivery_time'])]
actual_delivery_time.describe()
print('\n\nMean of actual time taken for delivery of products is: ' + str(actual_delivery_time.actual_delivery_time.mean()) + ' hr:mn:sec')



#_______________________________Sellers_data_______________________________________________________________

#Statewise seller ditribution


statewise_sellers = pd.DataFrame(sellers_data.seller_state.value_counts())
myplot1 = statewise_sellers.plot.bar(figsize=(10,10))
myplot1.set_ylabel('Number of unique sellers') 
myplot1.set_xlabel('States')
myplot1.set_title('Number of sellers per state')

    
    
    
    
    
    
    
    
    
#___________________________________________________________________________________________________________
#________________________________Merging dataset-----------------------------------------------------------

#Merging datasets

prod_value_1 = pd.merge(df_uni_product_price, df_uni_product, on='product_id') 
prod_value_2 = pd.merge(prod_value_1, product_data, on='product_id')
prod_value_3 = pd.merge(prod_value_2, df_product_freq, on='product_id')
prod_value_4 = pd.merge(order_review_data, order_item_data, on='order_id')

prod_value_6 = pd.merge(prod_value_3, df_prod_review, on='product_id')

for colname in prod_value_6:
    print(colname)

  

#___________________________________________________________________________________________________________
#________________________________Predicting product sales- ----------------------------------------------------------


cat = ["mean_product_cost","mean_product_fv","review_mean","product_weight_g","product_length_cm","product_height_cm","product_width_cm","no_of_products_sold"] 
prod_value = prod_value_6.loc[:,cat] 

prod_value_6.isnull().sum()

np.random.seed(0) 
nrows = len(prod_value)

randomlyShuffledRows = np.random.permutation(nrows)         #randomly shuffling rows
trainingRows = randomlyShuffledRows[0:26000]                  #26000 randomly selected rows to train 
testRows = randomlyShuffledRows[26000:]                       #Remaining rows for testing

xTrain = prod_value.iloc[trainingRows,0:7]
yTrain = prod_value.loc[trainingRows, 'no_of_products_sold']
xTest = prod_value.iloc[testRows,0:7]
yTest = prod_value.loc[testRows,'no_of_products_sold']


from sklearn import linear_model           #linear reggression
reg = linear_model.LinearRegression()

reg.fit(xTrain,yTrain)
reg.coef_ 
model_prediction = reg.predict(xTest)               #Prediction on the remaining rows of data.
diff = (model_prediction-yTest)






#___________________________________________________________________________________________________________
#________________________________Sales daywise-----------------------------------------------------------

orders_data.dtypes
orders_data['order_purchase_timestamp'] = pd.to_datetime(orders_data["order_purchase_timestamp"])    #convert column to datetime
orders_data['day_of_purchase'] = orders_data.order_purchase_timestamp.dt.weekday_name

day_sales = pd.DataFrame(orders_data.day_of_purchase.value_counts())      #counting sum of aech unique value of column

myplot = day_sales.plot.bar(figsize=(10,10))        #Plotting bar graph
myplot.set_ylabel('Number of purchases') 
myplot.set_xlabel('Day')
myplot.set_title('Daywise Sales')



#___________________________________________________________________________________________________________
#________________________________Category preffered-----------------------------------------------------------


product_data.isnull().sum()
product_data['product_category_name'].fillna(value= 'Others', inplace=True)

prod_category_count = pd.DataFrame(product_data.product_category_name.value_counts())      #counting sum of aech unique value of column
prod_category_count


#___________________________________________________________________________________________________________
#________________________________Statewise metrics-----------------------------------------------------------

statewise_1 = pd.merge(order_item_data, orders_data, on='order_id') 
statewise_1.isnull().sum()
statewise_2 = pd.merge(statewise_1, customer_data, on='customer_id')
statewise_2.isnull().sum()
statewise_3 = pd.merge(statewise_2, product_data, on='product_id')
statewise_3.isnull().sum()

unique_state = statewise_3.customer_state.unique()

statewise_t_price = []
statewise_t_freight = []
statewise_t_cost = []
statewise_m_freight = []
statewise_t_products_sold = []
statewise_product_cat_freq = []


len(unique_state)
len(statewise_t_price)
len(statewise_t_freight)
len(statewise_t_cost)
len(statewise_m_freight)
len(statewise_t_products_sold)

for i in range (len(unique_state)):
    temp = pd.DataFrame(statewise_3[statewise_3.customer_state == str(unique_state[i-1])])
    a = temp.price.sum()
    b = temp.freight_value.sum()
    c = temp.total_cost.sum()
    d = temp.freight_value.mean()
    e = temp.price.count()
    cat_freq = pd.DataFrame(temp.product_category_name.value_counts())                                         
    print('\n\nS.no- '+ str(i) +' : Top 3 popular product categories (with total products of that category sold) in the state ' + str(unique_state[i]) + ' are: \n' + str(cat_freq[0:3]) )
    
    statewise_t_price.append(a)
    statewise_t_freight.append(b)
    statewise_t_cost.append(c)
    statewise_m_freight.append(d)
    statewise_t_products_sold.append(e)
    statewise_product_cat_freq.append(cat_freq)
    i +=1


df_statewise = pd.DataFrame({'State':unique_state,'Total_products_sold':statewise_t_products_sold,'Total_products_price':statewise_t_price,'Total_freight_price':statewise_t_freight,'Total_product_cost':statewise_t_cost,'Mean_freight_cost':statewise_m_freight}, columns=['State','Total_products_sold','Total_products_price','Total_freight_price','Total_product_cost','Mean_freight_cost'])    #shows review mean of each unique product


print('\n\nState with maximum  mean freight cost is: \n' + str(df_statewise[df_statewise['Mean_freight_cost'] == df_statewise['Mean_freight_cost'].max()][['State','Mean_freight_cost']]))
print('\n\nState that sold maximum quantity of product is: \n' + str(df_statewise[df_statewise['Total_products_sold'] == df_statewise['Total_products_sold'].max()][['State','Total_products_sold']]))
print('\n\nState that had paid maximum freight cost is: \n' + str(df_statewise[df_statewise['Total_freight_price'] == df_statewise['Total_freight_price'].max()][['State','Total_freight_price']]))
print('\n\nState with maximum transactional amount is: \n' + str(df_statewise[df_statewise['Total_product_cost'] == df_statewise['Total_product_cost'].max()][['State','Total_product_cost']]))






#___________________________________________________________________________________________________________
#________________________________Correlations and heatmap-----------------------------------------------------------

prod_value_6['no_of_products_sold'].corr(prod_value_6['mean_product_cost'])
prod_value_6['no_of_products_sold'].corr(prod_value_6['mean_product_fv'])
prod_value_6['no_of_products_sold'].corr(prod_value_6['review_mean'])
prod_value_6['mean_product_cost'].corr(prod_value_6['review_mean'])
prod_value_6['mean_product_fv'].corr(prod_value_6['review_mean'])

from sklearn.decomposition import PCA
pca = PCA()
pca.fit(prod_value)

pca.components_
pca.explained_variance_

corrMatrix = prod_value_6.corr()
plt.figure(figsize=(12,9))
sns.heatmap(corrMatrix, annot=True)




#___________________________________________________________________________________________________________
#________________________________Line of best fit-----------------------------------------------------------

prod_value_6
prod_value_6.isnull().sum()


sd_d = prod_value_6['mean_product_fv'].std()
sd_p = prod_value_6['product_weight_g'].std()

prod_value_6.mean_product_fv.describe()
prod_value_6.product_weight_g.describe()

mean_d = prod_value_6['mean_product_fv'].mean()
mean_p = prod_value_6['product_weight_g'].mean()

cor = prod_value_6['mean_product_fv'].corr(prod_value_6['product_weight_g'])
slope = cor * sd_p/sd_d

print('\n The slope of the line of best fit between mean_product_fv and product_weight_g is: ' + str(slope))

intercept = mean_p-slope*mean_d
print('\n The y-intecept of the line of best fit is: ' + str(intercept))

#_____________________


sd_d = prod_value_6['mean_product_fv'].std()
sd_p = prod_value_6['product_height_cm'].std()
prod_value_6.mean_product_fv.describe()
prod_value_6.product_weight_g.describe()


mean_d = prod_value_6['mean_product_fv'].mean()
mean_p = prod_value_6['product_height_cm'].mean()

cor = prod_value_6['mean_product_fv'].corr(prod_value_6['product_height_cm'])
slope = cor * sd_p/sd_d

print('\n The slope of the line of best fit between mean_product_fv and product_height_cm is: ' + str(slope))

intercept = mean_p-slope*mean_d
print('\n The y-intecept of the line of best fit is: ' + str(intercept))

dia = np.linspace(0,400,10) 
plt.plot(dia,slope*dia + intercept,'-ok') 
plt.plot(prod_value_6['mean_product_fv'],prod_value_6['product_height_cm'],'ob')






#___________________________________________________________________________________________________________
#________________________________K-means clustering----------------------------------------------------------



geo_cust_1 = general_geolocation_data.rename(columns = {'geolocation_zip_code_prefix':'customer_zip_code_prefix'})
geo_customer_1 = pd.merge(geo_cust_1, customer_data, on='customer_zip_code_prefix',how='inner')
geo_customer = geo_customer_1.drop_duplicates('customer_unique_id')

geo_customer.isnull().sum()
len(geo_customer.customer_unique_id.unique())
geo_cust_final = geo_customer.iloc[:,1:3]

from sklearn import cluster
k = 8


kMeanResult = cluster.KMeans(k).fit(geo_cust_final)
kMeanResult.labels_
labelSymbols = ["*","+","o","s","^","x",".","p"] #give a different symbol for each cluster
labelColors = ['r','b','k','g','c','m','y','r'] #give a different color for each cluster

for i in range (len(geo_cust_final)):
    groupNumber = kMeanResult.labels_[i]
    symbol = labelSymbols[groupNumber]
    col = labelColors[groupNumber]
    plt.scatter(geo_cust_final.loc[i,"geolocation_lat"],geo_cust_final.loc[i,"geolocation_lng"],marker=symbol,c= col)
    plt.xlabel('')
    plt.ylabel('')










#___________________________________________________________________________________________________________
#________________________________line graph----------------------------------------------------------

orders_data
delivery_data = orders_data[pd.notnull(orders_data['order_delivered_customer_date'])]

orders_data['difference']
orders_data['actual_delivery_time'] = orders_data.loc[:,'order_delivered_customer_date'] - orders_data.loc[:,'order_purchase_timestamp']
orders_data['est_delivery_time'] = orders_data.loc[:,'order_estimated_delivery_date'] - orders_data.loc[:,'order_purchase_timestamp']
statewise_1.isnull().sum()

orders_data[pd.notnull(orders_data['actual_delivery_time'])]
orders_data['day_of_purchase'] = orders_data.order_purchase_timestamp.dt.weekday_name
j = orders_data.difference.dt.days
k = orders_data.est_delivery_time.dt.days
j

plot_actual = sns.distplot(j)
plot_estimated = sns.distplot(k)
plt.xlabel('Income / spend')
