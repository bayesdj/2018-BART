setwd("G:\\My Drive\\UT Austin\\BART Research\\BART Code")
#options(java.parameters = "-Xmx5000m")
library("bartMachine")
#set_bart_machine_num_cores(4)

df <- na.omit(read.csv('SkillCraft1_Dataset.csv',header=TRUE))
y <- df$APM
X <- df; X$APM <- NULL
n <- length(y)
X1 = X[1:2000,]
X2 = X[2001:n,]
y1 = y[1:2000]
y2 = y[2001:n]

ba = bartMachine(X1,y1,mh_prob_steps=c(2.5,2.5,4)/9,num_burn_in=250,num_iterations_after_burn_in=1000,num_trees=50)
yhat = predict(ba,X2)
e = yhat-y2
rmse = sqrt(e%*%e/length(e))

