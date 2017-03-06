p0 = 0.3;
l1 = 0.1;
l2 = 0.2;
p1 = 1-p0;
[trans,obs] = binchan(p0,l1,l0);
c_0_1 = 0;
c_1_0 = 0;
prob_1_0 = 0;
prob_0_1 = 0;
prob_false_alarm = 0;
prob_error = 0;

for n = 1:length(trans)
    if trans(n) == 1 & obs(n) == 0
        c_0_1 = c_0_1 + 1;
    end
end
        
prob_0_1 = (c_0_1)/ sum(trans==1);

for n = 1:length(trans)
    if trans(n)  == 0 & obs(n) == 1
        c_1_0 = c_1_0 + 1;
    end
end
        
prob_1_0 = (c_1_0) / sum(trans==0);

prob_error = (prob_1_0 * p0) + (prob_0_1 * p1);
prob_false_alarm = (c_1_0 * p0) / sum(trans==0);
fprintf('For p0 = %d, l1 = %d, and l2 = %d Probability of error = %d and Probability of False Alarm = %d\n',p0,l1,l2,prob_error, prob_false_alarm);
