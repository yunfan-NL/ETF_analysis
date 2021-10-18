from yahoo_fin.stock_info import *
import itertools
import matplotlib.pyplot as plt
import numpy as np

ticker = 'IWDA.AS'
#ticker = 'CNY.PA'

df_etf = get_data(ticker)
df_etf_monthly = df_etf.assign(Date=df_etf.index).resample('M',on='Date').mean()
df_etf_yearly = df_etf.assign(Date=df_etf.index).resample('Y',on='Date').mean()

# current month average compare with last month average
current_month_avg = df_etf_monthly.iloc[-1]['adjclose']
last_month_avg = df_etf_monthly.iloc[-2]['adjclose']
growth_rate_monthly_1 = (current_month_avg - last_month_avg)/last_month_avg

# current price compare with last month average
current_day_price = df_etf.iloc[-1]['adjclose']
growth_rate_monthly_2 = (current_day_price - last_month_avg)/last_month_avg

# add a column of the monthly growth rate
df_etf_monthly['growth_rate'] = df_etf_monthly.adjclose.diff()/df_etf_monthly['adjclose'].shift(1)
# add a column of the yearly growth rate
df_etf_yearly['growth_rate'] = df_etf_yearly.adjclose.diff()/df_etf_yearly['adjclose'].shift(1)

# if continuously decrease/increase for 3 month, calculate the growth rate
last_3rd_month_avg = df_etf_monthly.iloc[-4]['adjclose']
growth_rate_last_3rd = (current_day_price - last_3rd_month_avg)/last_3rd_month_avg

# if continuously decrease/increase for 2 month, calculate the growth rate
last_2nd_month_avg = df_etf_monthly.iloc[-3]['adjclose']
growth_rate_last_2nd = (current_day_price - last_2nd_month_avg)/last_2nd_month_avg

# print out the growth rates
print(ticker + '--- monthly growth rate (current month avg): '+str(growth_rate_monthly_1))
print(ticker + '--- monthly growth rate (todays price): '+ str(growth_rate_monthly_2))
print(ticker + '--- If there is continuous growth/decrease in 2 months, the growth rate is ' + str(growth_rate_last_2nd))
print(ticker + '--- If there is continuous growth/decrease in 3 months, the growth rate is ' + str(growth_rate_last_3rd))
df_etf_monthly.to_csv(ticker+'_monthly.csv')
df_etf.to_csv(ticker+'.csv')

#--------------------------------------------------------------------------
# calculate the frequency of continuous increase and decease month
growth_rate = df_etf_monthly['growth_rate'].values.tolist()
grm_pos = {}
grm_neg = {}
for k, v in itertools.groupby(growth_rate, lambda e:e>0):
    count = len(list(v))
    r = grm_pos
    if k == False:
        r = grm_neg
    if count not in r.keys():
        r[count] = 0
    r[count] += 1

# plot the frequency of continuous increase and decrease
fig, (axs1, axs2) = plt.subplots(1, 2)
fig.suptitle('Monthly Growth Histogram ('+ticker+')')
fig.set_size_inches(18.5, 10.5)
axs1.bar(grm_pos.keys(), grm_pos.values(), width=1, color='g')
axs1.set_title('Increase')
axs1.set(xlabel='Continuous increase',ylabel='Frequency')
x_pos = np.arange(1, max(grm_pos.keys())+1, 1)
y_pos = np.arange(1, max(grm_pos.values())+1,1)
axs1.set_xticks(x_pos)
axs1.set_yticks(y_pos)
axs2.bar(grm_neg.keys(), grm_neg.values(), width=1, color='r')
axs2.set_title('Decrease')
axs2.set(xlabel='Continuous decrease',ylabel='Frequency')
x_neg = np.arange(1, max(grm_neg.keys())+1, 1)
y_neg = np.arange(1, max(grm_neg.values())+1,1)
axs2.set_xticks(x_neg)
axs2.set_yticks(y_neg)
plt.savefig('Monthly growth histogram '+ticker+'.png')

#----------------------------------------------------------------
# calculate margin of continuous increase/decrease
# put column adjclose and growth rate to two lists
gr_m = df_etf_monthly['growth_rate'].values.tolist()
adjclose_m = df_etf_monthly['adjclose'].values.tolist()
# create two lists to store the calculated positive and negative margins
gr_margin_pos = []
gr_margin_neg = []
# for check only, can be deleted
# pointer=[]
# pointer1 = []
# j = 0
# for j in range(0,len(gr_m)):
#     if gr_m[j] > 0:
#         pointer.append(j)
# print(pointer)

# Find the pointers to calculate the positive growth rate. 
# Put in two lists p1: the start pointer, and p2: the end pointer
i = 0
p1 = []
for i in range(0, len(gr_m)-1):
    if gr_m[i] > 0 and i == 1:
        p1.append(i-1)
    elif gr_m[i] < 0 and (gr_m[i+1] > 0 ):
        p1.append(i)
        
p2 = []
for i in range(0, len(gr_m)-1):
    if gr_m[i] > 0 and gr_m[i+1] < 0:
        p2.append(i)
    elif gr_m[i] > 0 and gr_m[i+1] > 0 and i == len(gr_m)-2:
         p2.append(i+1)
    if gr_m[i+1] > 0 and i == len(gr_m)-2:
        p2.append(i+1)
    i += 1
print("Margin calculation check---------------------------------------------")
print(p1)
print(p2)
# Calculate the positive margin, use the pointer to find the relative price in adjclose list
i = 0
for i in range(0, len(p1)):
    margin = (adjclose_m[p2[i]] - adjclose_m[p1[i]])/adjclose_m[p1[i]]
    gr_margin_pos.append(margin)
#print(gr_margin_pos)

# for check only
# pointer=[]
# pointer1 = []
# j = 0
# for j in range(0,len(gr_m)):
#     if gr_m[j] < 0:
#         pointer.append(j)
# print(pointer)

# Find the pointers to calculate the negative growth rate. 
# Put in two lists p3: the start pointer, and p4: the end pointer
i = 0
p3 = []
for i in range(0, len(gr_m)-1):
    if gr_m[i] < 0 and i == 1:
        p3.append(i-1)
    elif gr_m[i] > 0 and gr_m[i+1] < 0:
        p3.append(i)
    i += 1

p4 = []
for i in range(0, len(gr_m)-1):
    if gr_m[i] < 0 and gr_m[i+1] > 0:
        p4.append(i)
    elif gr_m[i] < 0 and gr_m[i+1] < 0 and i == len(gr_m)-2:
        p4.append(i+1)
    elif gr_m[i] > 0 and gr_m[i+1] < 0 and i == len(gr_m)-2:
        p4.append(i+1)
    i += 1
print(p3)
print(p4)

# Calculate the negative margin
i = 0
for i in range(0, len(p3)):
    margin = (adjclose_m[p4[i]] - adjclose_m[p3[i]])/adjclose_m[p3[i]]
    gr_margin_neg.append(margin)
print(gr_margin_neg)

# count growth rate positive in ranges
i1 = 0 # between 0-0.05
i2 = 0 # between 0.05-0.1
i3 = 0 # between 0.1-0.15
i4 = 0 # between 0.15-0.2
i5 = 0 # between 0.2-0.25
i6 = 0 # between 0.25-0.3
i7 = 0 # between 0.3-0.35
i8 = 0 # between 0.35-0.4
i9 = 0 # between 0.4-0.45
i10 = 0 # between 0.45-0.5
i11 = 0 # between 0.5-0.55
i12 = 0 # between 0.55-0.6
i13 = 0 # between 0.6-0.65
i14 = 0 # between 0.65-0.7
i15 = 0 # between 0.7-0.75
i16 = 0 # between 0.75-0.8
i17 = 0 # between 0.8-0.85
i18 = 0 # between 0.85-0.9
i19 = 0 # between 0.9-0.95
i20 = 0 # between 0.95-1

for i in range(0,len(gr_margin_pos)-1):
    if gr_margin_pos[i]<=0.05:
        i1+=1
    elif 0.05<gr_margin_pos[i]<=0.1:
        i2+=1
    elif 0.1<gr_margin_pos[i]<=0.15:
        i3+=1
    elif 0.15<gr_margin_pos[i]<=0.2:
        i4+=1
    elif 0.2<gr_margin_pos[i]<=0.25:
        i5+=1
    elif 0.25<gr_margin_pos[i]<=0.3:
        i6+=1
    elif 0.3<gr_margin_pos[i]<=0.35:
        i7+=1
    elif 0.35<gr_margin_pos[i]<=0.4:
        i8+=1
    elif 0.4<gr_margin_pos[i]<=0.45:
        i9+=1
    elif 0.45<gr_margin_pos[i]<=0.5:
        i10+=1
    elif 0.5<gr_margin_pos[i]<=0.55:
        i11+=1
    elif 0.55<gr_margin_pos[i]<=0.6:
        i12+=1
    elif 0.6<gr_margin_pos[i]<=0.65:
        i13+=1
    elif 0.65<gr_margin_pos[i]<=0.7:
        i14+=1
    elif 0.7<gr_margin_pos[i]<=0.75:
        i15+=1
    elif 0.75<gr_margin_pos[i]<=0.8:
        i16+=1
    elif 0.8<gr_margin_pos[i]<=0.85:
        i17+=1
    elif 0.85<gr_margin_pos[i]<=0.9:
        i18+=1
    elif 0.9<gr_margin_pos[i]<=0.95:
        i19+=1
    else:
        i20+=1
    i+=1
count_gr_margin_pos = [i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13,i14,i15,i16,i17,i18,i19,i20]
print(count_gr_margin_pos)

# count growth rate negative between ranges
i1 = 0 # between 0-0.05
i2 = 0 # between 0.05-0.1
i3 = 0 # between 0.1-0.15
i4 = 0 # between 0.15-0.2
i5 = 0 # between 0.2-0.25
i6 = 0 # between 0.25-0.3
i7 = 0 # between 0.3-0.35
i8 = 0 # between 0.35-0.4
i9 = 0 # between 0.4-0.45
i10 = 0 # between 0.45-0.5
i11 = 0 # between 0.5-0.55
i12 = 0 # between 0.55-0.6
i13 = 0 # between 0.6-0.65
i14 = 0 # between 0.65-0.7
i15 = 0 # between 0.7-0.75
i16 = 0 # between 0.75-0.8
i17 = 0 # between 0.8-0.85
i18 = 0 # between 0.85-0.9
i19 = 0 # between 0.9-0.95
i20 = 0 # between 0.95-1

for i in range(0,len(gr_margin_neg)-1):
    if gr_margin_neg[i]>-0.05:
        i1+=1
    elif -0.1<gr_margin_neg[i]<=0.05:
        i2+=1
    elif -0.15<gr_margin_neg[i]<=0.1:
        i3+=1
    elif -0.2<gr_margin_neg[i]<=-0.15:
        i4+=1
    elif -0.25<gr_margin_neg[i]<=-0.2:
        i5+=1
    elif -0.3<gr_margin_neg[i]<=-0.25:
        i6+=1
    elif -0.35<gr_margin_neg[i]<=-0.3:
        i7+=1
    elif -0.4<gr_margin_neg[i]<=-0.35:
        i8+=1
    elif -0.45<gr_margin_neg[i]<=-0.4:
        i9+=1
    elif -0.5<gr_margin_neg[i]<=-0.45:
        i10+=1
    elif -0.55<gr_margin_neg[i]<=-0.5:
        i11+=1
    elif -0.5<gr_margin_neg[i]<=-0.55:
        i12+=1
    elif -0.65<gr_margin_neg[i]<=-0.6:
        i13+=1
    elif -0.7<gr_margin_neg[i]<=-0.65:
        i14+=1
    elif -0.75<gr_margin_neg[i]<=-0.7:
        i15+=1
    elif -0.8<gr_margin_neg[i]<=-0.75:
        i16+=1
    elif -0.85<gr_margin_neg[i]<=-0.8:
        i17+=1
    elif -0.9<gr_margin_neg[i]<=-0.85:
        i18+=1
    elif -0.95<gr_margin_neg[i]<=-0.9:
        i19+=1
    else:
        i20+=1
    i+=1
count_gr_margin_neg = [i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13,i14,i15,i16,i17,i18,i19,i20]
print(count_gr_margin_neg)

# plot the count growth margin both positive and negative
# put the count to two dictionaries
count_mg_pos = {}
i=0
for i in range(0,20,1):
    count_mg_pos[(i+1)*5] = count_gr_margin_pos[i]
    i+=1
count_mg_neg = {}
i=0
for i in range(0,20,1):
    count_mg_neg[(i+1)*5] = count_gr_margin_neg[i]
    i+=1
print(count_mg_pos)
print(count_mg_neg)
fig, (axs1, axs2) = plt.subplots(1, 2)
fig.suptitle('Monthly Growth Margin Distribution Histogram ('+ticker+')')
fig.set_size_inches(18.5, 10.5)
axs1.bar(count_mg_pos.keys(), count_mg_pos.values(), width=-5, color='g',align='edge')
axs1.set_title('Increase')
axs1.set(xlabel='Continuous increase margin (<=)',ylabel='Frequency')
x_pos = np.arange(0, max(count_mg_pos.keys())+5, 5)
y_pos = np.arange(1, max(count_mg_pos.values())+1,1)
axs1.set_xticks(x_pos)
axs1.set_yticks(y_pos)
axs2.bar(count_mg_neg.keys(), count_mg_neg.values(), width=-5, color='r',align='edge')
axs2.set_title('Decrease')
axs2.set(xlabel='Continuous decrease margin (<=)',ylabel='Frequency')
x_neg = np.arange(0, max(count_mg_neg.keys())+5, 5)
y_neg = np.arange(1, max(count_mg_neg.values())+1,1)
axs2.set_xticks(x_neg)
axs2.set_yticks(y_neg)
plt.savefig("Monthly growth margin distribution "+ticker+'.png')