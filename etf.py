from yahoo_fin.stock_info import *
import itertools
import matplotlib.pyplot as plt
import numpy as np

def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

def adjust_lightness(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

ticker = 'IWDA.AS'
#ticker = 'CNY.PA'
#ticker = 'BNK.PA'

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
# add a column of the 3 monthly growth rate
df_etf_monthly['3_months_growth_rate'] = df_etf_monthly.adjclose.diff(periods=3)/df_etf_monthly['adjclose'].shift(3)
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
axs1.set(xlabel='Continuous number of month that increase',ylabel='Frequency')
x_pos = np.arange(1, max(grm_pos.keys())+1, 1)
y_pos = np.arange(1, max(grm_pos.values())+1,1)
axs1.set_xticks(x_pos)
axs1.set_yticks(y_pos)
axs2.bar(grm_neg.keys(), grm_neg.values(), width=1, color='r')
axs2.set_title('Decrease')
axs2.set(xlabel='Continuous number of month that decrease',ylabel='Frequency')
x_neg = np.arange(1, max(grm_neg.keys())+1, 1)
y_neg = np.arange(1, max(grm_neg.values())+1,1)
axs2.set_xticks(x_neg)
axs2.set_yticks(y_neg)
plt.savefig('Countinuous monthly growth histogram '+ticker+'.png')

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
#print(gr_margin_neg)

# count growth rate positive in ranges
count_mg_pos = {}
count = 0
interval = 1
for interval in range(0,100):
    for j in range(0,len(gr_margin_pos)):
        if interval*0.01<gr_margin_pos[j]<(interval+1)*0.01:
            count+=1
    count_mg_pos[interval] = count
    count = 0

# count growth rate negative between ranges
count_mg_neg = {}
count = 0
interval = 1
for interval in range(0,100):
    for j in range(0,len(gr_margin_neg)):
        if -interval*0.01>gr_margin_neg[j]>=-(interval+1)*0.01:
            count+=1
    count_mg_neg[interval] = count
    count = 0

# plot the count growth margin both positive and negative
fig, (axs1, axs2) = plt.subplots(2, 1)
fig.suptitle('Monthly Growth Margin Distribution Histogram ('+ticker+')')
fig.set_size_inches(25, 10.5)
axs1.bar(count_mg_pos.keys(), count_mg_pos.values(), width=1, color='g',align='edge')
axs1.set_title('Increase')
axs1.set(xlabel='Continuous increase margin in % (<=)',ylabel='Frequency')
x_pos = np.arange(0, max(count_mg_pos.keys())+1, 1)
y_pos = np.arange(1, max(count_mg_pos.values())+1,1)
axs1.set_xticks(x_pos)
axs1.set_yticks(y_pos)
axs2.bar(count_mg_neg.keys(), count_mg_neg.values(), width=1, color='r',align='edge')
axs2.set_title('Decrease')
axs2.set(xlabel='Continuous decrease margin in % (<=)',ylabel='Frequency')
x_neg = np.arange(0, max(count_mg_neg.keys())+1, 1)
y_neg = np.arange(1, max(count_mg_neg.values())+1,1)
axs2.set_xticks(x_neg)
axs2.set_yticks(y_neg)
plt.savefig("Monthly growth margin distribution "+ticker+'.png')

#--------------------------------------------------------------------------
# plot the distribution/histogram of the grow/decrease
growth_rate = df_etf_monthly['growth_rate'].values.tolist()
pos = []
neg = []
for i in range(0,len(growth_rate)):
    if growth_rate[i] > 0:
        pos.append(growth_rate[i])
    else:
        neg.append(growth_rate[i])

count_grm_pos = {}
count = 0
interval = 1
for interval in range(0,100):
    for j in range(0,len(pos)):
        if interval*0.01<pos[j]<(interval+1)*0.01:
            count+=1
    count_grm_pos[interval] = count
    count = 0

count_grm_neg = {}
count = 0
interval = 1
for interval in range(0,100):
    for k in range(0,len(neg)):
        if -interval*0.01>neg[k]>-(interval+1)*0.01:
            count+=1
    count_grm_neg[interval] = count
    count = 0

# plot the frequency of increase and decrease
fig, (axs1, axs2) = plt.subplots(nrows=2, ncols=1)
fig.suptitle('Monthly Growth Histogram ('+ticker+')')
fig.set_size_inches(25, 10.5)
axs1.bar(count_grm_pos.keys(), count_grm_pos.values(), width=1, color='g',align='edge')
axs1.set_title('Increase')
axs1.set(xlabel='Monthly increase margin in % (<=)',ylabel='Frequency')
x_pos = np.arange(0, max(count_grm_pos.keys())+1, 1)
y_pos = np.arange(1, max(count_grm_pos.values())+1,1)
axs1.set_xticks(x_pos)
axs1.set_yticks(y_pos)
axs2.bar(count_grm_neg.keys(), count_grm_neg.values(), width=1, color='r',align='edge')
axs2.set_title('Decrease')
axs2.set(xlabel='Month decrease marge in % (<=)',ylabel='Frequency')
x_neg = np.arange(0, max(count_grm_neg.keys())+1, 1)
y_neg = np.arange(1, max(count_grm_neg.values())+1,1)
axs2.set_xticks(x_neg)
axs2.set_yticks(y_neg)
plt.savefig('Monthly growth histogram '+ticker+'.png')

# --------------------------------------------------------
# calculate the probabilities in the category of increase:
in_in = 0
in_de = 0
for i in range(0,len(growth_rate)-1):
    if growth_rate[i] > 0 and growth_rate[i+1] > 0:
        in_in += 1
    elif growth_rate[i] > 0 and growth_rate[i+1]<0:
        in_de += 1
p_in_in = in_in/(in_in + in_de)
p_in_de = 1 - p_in_in
print("increase-increase %: "+str(p_in_in))
print("increase-decrease %: "+str(p_in_de))

in_in_list = []
in_de_list = []
mg = 0
for k in range(0,100):
    count_in = 0
    count_de = 0
    for i in range(0,len(growth_rate)-1):
        if growth_rate[i] >= mg and growth_rate[i] < mg+0.01:
            if growth_rate[i+1] > 0:
                count_in += 1
            elif growth_rate[i+1] < 0:
                count_de += 1
    in_in_list.append(count_in)
    in_de_list.append(count_de)
    mg += 0.01

raw_data = {'in_in':in_in_list,'in_de':in_de_list}
df = pd.DataFrame(raw_data)
totals = [i+j for i,j in zip(df['in_in'], df['in_de'])]
in_in_percent = [round(i / j * 100) if j != 0 else 0 for i,j in zip(df['in_in'], totals)]
in_de_percent = [round(i / j * 100) if j != 0 else 0 for i,j in zip(df['in_de'], totals)]    
#print(in_in_percent)
#print(in_de_percent)

# calculate the probabilities in the category of decrease:
de_in = 0
de_de = 0
for i in range(0,len(growth_rate)-1):
    if growth_rate[i] <0 and growth_rate[i+1] > 0:
        de_in += 1
    elif growth_rate[i] < 0 and growth_rate[i+1]<0:
        de_de += 1
p_de_in = de_in/(de_in + de_de)
p_de_de = 1 - p_de_in
print("decrease-increase %: "+str(p_de_in))
print("decrease-increase %: "+str(p_de_de))

de_in_list = []
de_de_list = []
mg = 0
for k in range(0,100):
    count_in = 0
    count_de = 0
    for i in range(0,len(growth_rate)-1):
        if growth_rate[i] < -mg and growth_rate[i] >= -(mg+0.01):
            if growth_rate[i+1] > 0:
                count_in += 1
            elif growth_rate[i+1] < 0:
                count_de += 1
    de_in_list.append(count_in)
    de_de_list.append(count_de)
    mg += 0.01
# from raw value to percentage
raw_data = {'de_in':de_in_list,'de_de':de_de_list}
df = pd.DataFrame(raw_data)
totals_de = [i+j for i,j in zip(df['de_in'], df['de_de'])]
de_in_percent = [round(i / j * 100) if j != 0 else 0 for i,j in zip(df['de_in'], totals_de)]
de_de_percent = [round(i / j * 100) if j != 0 else 0 for i,j in zip(df['de_de'], totals_de)]    
#print(de_in_percent)
#print(de_de_percent)

#plot both increase and decrease category
r = range(0,100,1)
name = np.arange(0, len(in_in_percent)+1, 1)
fig, (axs1, axs2) = plt.subplots(nrows=2, ncols=1)
fig.suptitle('The chance of increase/decrease for the next month, if this month increase/decrease with certain % ('+ticker+')')
fig.set_size_inches(25, 10.5)
axs1.bar(r,in_in_percent,color='#b5ffb9',edgecolor='white',width=1,label='increase-increase',align='edge')
axs1.bar(r,in_de_percent,bottom=in_in_percent,color='#f9bc86',edgecolor='white',width=1,label='increase-decrease',align='edge')
axs1.set_title('This month increase')
axs1.set_xticks(name)
axs1.xaxis.set_tick_params(rotation=90)
axs1.set_xlabel('increase rate of this month')
axs1.set_ylabel('the chance of increase/decrease in %')
axs1.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
axs2.bar(r,de_in_percent,color='#b5ffb9',edgecolor='white',width=1,label='decrease-increase',align='edge')
axs2.bar(r,de_de_percent,bottom=de_in_percent,color='#f9bc86',edgecolor='white',width=1,label='decrease-decrease',align='edge')
axs2.set_title('This month decrease')
axs2.set_xticks(name)
axs2.xaxis.set_tick_params(rotation=90)
axs2.set_xlabel('decrease rate of this month')
axs2.set_ylabel('the chance of increase/decrease in %')
axs2.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)

y_offset=-3
for bar in axs1.patches:
  axs1.text(
      # Put the text in the middle of each bar. get_x returns the start
      # so we add half the width to get to the middle.
      bar.get_x() + bar.get_width() / 2,
      # Vertically, add the height of the bar to the start of the bar,
      # along with the offset.
      bar.get_height() + bar.get_y() + y_offset,
      # This is actual value we'll show.
      round(bar.get_height()),
      # Center the labels and style them a bit.
      ha='center',
      color='black',
      size=6
  )
for bar in axs2.patches:
  axs2.text(
      # Put the text in the middle of each bar. get_x returns the start
      # so we add half the width to get to the middle.
      bar.get_x() + bar.get_width() / 2,
      # Vertically, add the height of the bar to the start of the bar,
      # along with the offset.
      bar.get_height() + bar.get_y() + y_offset,
      # This is actual value we'll show.
      round(bar.get_height()),
      # Center the labels and style them a bit.
      ha='center',
      color='black',
      size=6
  )
plt.savefig('The chance of in or de next month, if this month in or de with certain percentage ('+ticker+').png')

#----------------------------------------------------------------------------------------------
# the previous 3 months on average increase or decease, then the chance of increase/decrease next month
# calculate the probabilities in the category of increase:
growth_rate_3 = df_etf_monthly['3_months_growth_rate'].values.tolist()
growth_rate_3 = [x for x in growth_rate_3 if str(x) != 'nan']
in_in = 0
in_de = 0
for i in range(0,len(growth_rate_3)-1):
    if growth_rate_3[i] > 0 and growth_rate_3[i+1] > 0:
        in_in += 1
    elif growth_rate_3[i] > 0 and growth_rate_3[i+1]<0:
        in_de += 1
p_in_in = in_in/(in_in + in_de)
p_in_de = 1 - p_in_in
print("compare with 3 months ago increase-increase %: "+str(p_in_in))
print("compare with 3 months ago increase-decrease %: "+str(p_in_de))

in_in_list = []
in_de_list = []
mg = 0
for k in range(0,100):
    count_in = 0
    count_de = 0
    for i in range(0,len(growth_rate_3)-1):
        if growth_rate_3[i] >= mg and growth_rate_3[i] < mg+0.01:
            if growth_rate_3[i+1] > 0:
                count_in += 1
            elif growth_rate_3[i+1] < 0:
                count_de += 1
    in_in_list.append(count_in)
    in_de_list.append(count_de)
    mg += 0.01

raw_data = {'in_in':in_in_list,'in_de':in_de_list}
df = pd.DataFrame(raw_data)
totals = [i+j for i,j in zip(df['in_in'], df['in_de'])]
in_in_percent = [round(i / j * 100) if j != 0 else 0 for i,j in zip(df['in_in'], totals)]
in_de_percent = [round(i / j * 100) if j != 0 else 0 for i,j in zip(df['in_de'], totals)]    
#print(in_in_percent)
#print(in_de_percent)

# calculate the probabilities in the category of decrease:
de_in = 0
de_de = 0
for i in range(0,len(growth_rate_3)-1):
    if growth_rate_3[i] <0 and growth_rate_3[i+1] > 0:
        de_in += 1
    elif growth_rate_3[i] < 0 and growth_rate_3[i+1]<0:
        de_de += 1
p_de_in = de_in/(de_in + de_de)
p_de_de = 1 - p_de_in
print("compare with 3 months ago decrease-increase %: "+str(p_de_in))
print("compare with 3 months ago decrease-increase %: "+str(p_de_de))

de_in_list = []
de_de_list = []
mg = 0
for k in range(0,100):
    count_in = 0
    count_de = 0
    for i in range(0,len(growth_rate_3)-1):
        if growth_rate_3[i] < -mg and growth_rate_3[i] >= -(mg+0.01):
            if growth_rate_3[i+1] > 0:
                count_in += 1
            elif growth_rate_3[i+1] < 0:
                count_de += 1
    de_in_list.append(count_in)
    de_de_list.append(count_de)
    mg += 0.01
# from raw value to percentage
raw_data = {'de_in':de_in_list,'de_de':de_de_list}
df = pd.DataFrame(raw_data)
totals_de = [i+j for i,j in zip(df['de_in'], df['de_de'])]
de_in_percent = [round(i / j * 100) if j != 0 else 0 for i,j in zip(df['de_in'], totals_de)]
de_de_percent = [round(i / j * 100) if j != 0 else 0 for i,j in zip(df['de_de'], totals_de)]    
#print(de_in_percent)
#print(de_de_percent)


#plot both increase and decrease category
r = range(0,100,1)
name = np.arange(0, len(in_in_percent)+1, 1)
fig, (axs1, axs2) = plt.subplots(nrows=2, ncols=1)
fig.suptitle('The chance of increase/decrease for the next month, if compare with 3 months ago increase/decrease with certain % ('+ticker+')')
fig.set_size_inches(25, 10.5)
axs1.bar(r,in_in_percent,color='#b5ffb9',edgecolor='white',width=1,label='increase-increase',align='edge')
axs1.bar(r,in_de_percent,bottom=in_in_percent,color='#f9bc86',edgecolor='white',width=1,label='increase-decrease',align='edge')
axs1.set_title('This month increase')
axs1.set_xticks(name)
axs1.xaxis.set_tick_params(rotation=90)
axs1.set_xlabel('increase rate compare with 3 months ago')
axs1.set_ylabel('the chance of increase/decrease in %')
axs1.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
axs2.bar(r,de_in_percent,color='#b5ffb9',edgecolor='white',width=1,label='decrease-increase',align='edge')
axs2.bar(r,de_de_percent,bottom=de_in_percent,color='#f9bc86',edgecolor='white',width=1,label='decrease-decrease',align='edge')
axs2.set_title('This month decrease')
axs2.set_xticks(name)
axs2.xaxis.set_tick_params(rotation=90)
axs2.set_xlabel('decrease rate compare with 3 months ago')
axs2.set_ylabel('the chance of increase/decrease in %')
axs2.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)

y_offset=-3
for bar in axs1.patches:
  axs1.text(
      # Put the text in the middle of each bar. get_x returns the start
      # so we add half the width to get to the middle.
      bar.get_x() + bar.get_width() / 2,
      # Vertically, add the height of the bar to the start of the bar,
      # along with the offset.
      bar.get_height() + bar.get_y() + y_offset,
      # This is actual value we'll show.
      round(bar.get_height()),
      # Center the labels and style them a bit.
      ha='center',
      color='black',
      size=6
  )
for bar in axs2.patches:
  axs2.text(
      # Put the text in the middle of each bar. get_x returns the start
      # so we add half the width to get to the middle.
      bar.get_x() + bar.get_width() / 2,
      # Vertically, add the height of the bar to the start of the bar,
      # along with the offset.
      bar.get_height() + bar.get_y() + y_offset,
      # This is actual value we'll show.
      round(bar.get_height()),
      # Center the labels and style them a bit.
      ha='center',
      color='black',
      size=6
  )
plt.savefig('The chance of in or de next month, if compare with 3 months in or de with certain percentage ('+ticker+').png')