% Load the data and plot the histogram for distribution
data = 'hw3dat';
file = readtable(data);
t = table2array(file);

%Prior Calculation

p0 = sum(t(:,3) == 0)/length(t(:,3));
p1 = 1-p0;

% Histogram Plotting

subplot(1,2,1)
hist(t(:,1));
title('Histogram Plotting for H0')
xlabel('H0')

subplot(1,2,2)
hist(t(:,2));
title('Histogram Plotting for H1')
xlabel('H1')

%Density Plotting

subplot(1,2,1)
ksdensity(t(:,1));
title(‘Density Plotting for H0')
xlabel('H0')

subplot(1,2,2)
ksdensity(t(:,2));
title('Density Plotting for H1')
xlabel('H1')

% After examining histogram and density plot I concluded following
% decision rules

% First Rule
% H0 = 0 if value of y is between -1 to 1 otherwise H1=1
% Calculation for first decision rule

dec_value_zero = t(:,4)>=-1 & t(:,4)<=1;
dec_value_zero = ~dec_value_zero;

% Calculate the cost
c00 = sum(t(:,3)== 0 & dec_value_zero == 0)/length(t(:,3));
c01 = sum(t(:,3)== 1 & dec_value_zero == 0)/length(t(:,3));
c11 = sum(t(:,3)== 1 & dec_value_zero == 1)/length(t(:,3));
c10 = sum(t(:,3)== 0 & dec_value_zero == 1)/length(t(:,3));

% Taking uniform cost to compute Baye's risk for the first rule
bayes_r_1 = p0*(sum(t(:,3)== 0 & dec_value_zero == 1)/length(t))+ p1*(sum(t(:,3)== 1 & dec_value_zero == 0)/length(t))

% Second rule
% H0 = 0 if value of y is between -1.5 to 1.5 otherwise H1=1
% Calculation for second decision rule

dec_value_zero = t(:,4)>=-1.5 & t(:,4)<=1.5;
dec_value_zero = ~dec_value_zero;

% Calculate the cost
c00 = sum(t(:,3)== 0 & dec_value_zero == 0)/length(t(:,3));
c01 = sum(t(:,3)== 1 & dec_value_zero == 0)/length(t(:,3));
c11 = sum(t(:,3)== 1 & dec_value_zero == 1)/length(t(:,3));
c10 = sum(t(:,3)== 0 & dec_value_zero == 1)/length(t(:,3));

% Taking uniform cost and computing Baye's risk for the second rule
bayes_risk_2 = p0*(sum(t(:,3)== 0 & dec_value_zero == 1)/length(t))+ p1*(sum(t(:,3)== 1 & dec_value_zero == 0)/length(t))