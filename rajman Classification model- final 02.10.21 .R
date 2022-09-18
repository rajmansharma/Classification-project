################################
###  CASE STUDY ANALSYSIS
#################################
##### Importing data
data1=read.csv("D:/Upgrad/Classification/heart_failure_clinical_records_dataset.csv")
df=data1
##### Checking number of observations (customers) on whom the analysis will be performed 
N=nrow(data1)
N

nn=nrow(data1)


## Data size and structure
library(dplyr)
glimpse(df)

##considering following featured for data visualization.
f_features = c("anaemia", "diabetes", "high_blood_pressure", "sex", "smoking", "DEATH_EVENT")


##data visualisation.
library(ggplot2)
plot1= ggplot(df, aes(x = anaemia, fill = DEATH_EVENT)) +
  geom_bar(stat = "count", position = "stack", show.legend = FALSE) +
  scale_x_discrete(labels  = c("0 (False)", "1 (True)"))+
  scale_fill_manual(values = c("blue","green"),
                    name = "DEATH_EVENT",
                    labels = c("0 (False)", "1 (True)")) +
  labs(x = "Anaemia") +
  theme_minimal(base_size = 12) +
  geom_label(stat = "count", aes(label = ..count..), position = position_stack(vjust = 0.5),
             size = 5, show.legend = FALSE)
plot(plot1)

plot2= ggplot(df, aes(x = diabetes, fill = DEATH_EVENT)) +
  geom_bar(stat = "count", position = "stack", show.legend = FALSE) +
  scale_x_discrete(labels  = c("0 (False)", "1 (True)"))+
  scale_fill_manual(values = c("blue","green"),
                    name = "DEATH_EVENT",
                    labels = c("0 (False)", "1 (True)")) +
  labs(x = "diabetes") +
  theme_minimal(base_size = 12) +
  geom_label(stat = "count", aes(label = ..count..), position = position_stack(vjust = 0.5),
             size = 5, show.legend = FALSE)
plot(plot2)


plot3= ggplot(df, aes(x = high_blood_pressure, fill = DEATH_EVENT)) +
  geom_bar(stat = "count", position = "stack", show.legend = FALSE) +
  scale_x_discrete(labels  = c("0 (False)", "1 (True)"))+
  scale_fill_manual(values = c("blue","green"),
                    name = "DEATH_EVENT",
                    labels = c("0 (False)", "1 (True)")) +
  labs(x = "high_blood_pressure") +
  theme_minimal(base_size = 12) +
  geom_label(stat = "count", aes(label = ..count..), position = position_stack(vjust = 0.5),
             size = 5, show.legend = FALSE)
plot(plot3)


plot4= ggplot(df, aes(x = sex, fill = DEATH_EVENT)) +
  geom_bar(stat = "count", position = "stack", show.legend = FALSE) +
  scale_x_discrete(labels  = c("0 (False)", "1 (True)"))+
  scale_fill_manual(values = c("blue","green"),
                    name = "DEATH_EVENT",
                    labels = c("0 (False)", "1 (True)")) +
  labs(x = "sex ") +
  theme_minimal(base_size = 12) +
  geom_label(stat = "count", aes(label = ..count..), position = position_stack(vjust = 0.5),
             size = 5, show.legend = FALSE)
plot(plot4)

plot5= ggplot(df, aes(x = smoking, fill = DEATH_EVENT)) +
  geom_bar(stat = "count", position = "stack", show.legend = FALSE) +
  scale_x_discrete(labels  = c("0 (False)", "1 (True)"))+
  scale_fill_manual(values = c("blue","green"),
                    name = "DEATH_EVENT",
                    labels = c("0 (False)", "1 (True)")) +
  labs(x = "smoking ") +
  theme_minimal(base_size = 12) +
  geom_label(stat = "count", aes(label = ..count..), position = position_stack(vjust = 0.5),
             size = 5, show.legend = FALSE)
plot(plot5)




plot6= ggplot(df, aes(x = DEATH_EVENT, fill = DEATH_EVENT)) +
  geom_bar(stat = "count", position = "stack", show.legend = FALSE) +
  scale_x_discrete(labels  = c("0 (False)", "1 (True)"))+
  scale_fill_manual(values = c("blue","green"),
                    name = "DEATH_EVENT",
                    labels = c("0 (False)", "1 (True)")) +
  labs(x = "DEATH_EVENT ") +
  theme_minimal(base_size = 12) +
  geom_label(stat = "count", aes(label = ..count..), position = position_stack(vjust = 0.5),
             size = 5, show.legend = FALSE)
plot(plot6)

## perform Shapiro-Wilk's test for checking the normality of dataset:
shapiro.test(data1$age)
shapiro.test(data1$anaemia)
shapiro.test(data1$creatinine_phosphokinase)
shapiro.test(data1$diabetes)
shapiro.test(data1$ejection_fraction)
shapiro.test(data1$high_blood_pressure)
shapiro.test(data1$platelets)
shapiro.test(data1$serum_creatinine)
shapiro.test(data1$serum_sodium)
shapiro.test(data1$sex)
shapiro.test(data1$smoking)
shapiro.test(data1$time)

##Perform two-sample t-tests on predictors 

library(ggpubr)
t.test(age~anaemia, data=df)
t.test(serum_creatinine ~anaemia, data=df)
t.test(time~anaemia, data=df)
t.test(platelets~anaemia, data=df)

### Creating training and test set
set.seed(123)
indx=sample(1:nn,0.8*nn)
traindata=data1[indx,]
testdata=data1[-indx,]



#### Fitting full logistic regression (LR) model with all features
fullmod=glm(as.factor(DEATH_EVENT)~as.factor(anaemia)
            +creatinine_phosphokinase+age+as.factor(diabetes)+ejection_fraction+as.factor(high_blood_pressure)
+platelets+serum_creatinine+serum_sodium+as.factor(sex)+as.factor(smoking)+time,data=traindata,family="binomial")

summary(fullmod)


#### Selecting features for fitting reduced logistic regression model
library(MASS)
step=stepAIC(fullmod)

mod2=glm(as.factor(DEATH_EVENT)~age+ejection_fraction+serum_creatinine+time, family = "binomial", 
         data = traindata)
summary(mod2)

### predicting success probabilities using the LR model
testdata_new=testdata[,c(1,5,8,12)]
pred_prob=predict(mod2,testdata_new,type="response")
hist(pred_prob)

### predicting success probability for an individual

library(stats)
library(datasets)
sampletest=data.frame(t(c(50,50,2.0,60)))
colnames(sampletest)=c("age","ejection_fraction","serum_creatinine","time")
predict(mod2,sampletest,type="response")


#### Plotting ROC 
library(pROC)
roc1=roc(testdata[,13],pred_prob,plot=TRUE,legacy.axes=TRUE)
plot(roc1)
roc1$auc


#### Using ROC in deciding threshold
library(lattice)
library(ggplot2)
thres=data.frame(sen=roc1$sensitivities, spec=roc1$specificities,thresholds=roc1$thresholds)
thres[thres$sen>0.70&thres$spec>0.4,]

pred_Y=ifelse(pred_prob > 0.093,1,0)
confusionMatrix(as.factor(testdata[,13]), as.factor(pred_Y))



###############################
## Random Forest
###############################

set.seed(0)
library(ranger)

df_new2=df
library(randomForest)
df_new2$age=(df_new2$age)
df_new2$ejection_fraction=(df_new2$ejection_fraction)
df_new2$serum_creatinine=(df_new2$serum_creatinine)
df_new2$time=(df_new2$time)
df_new2$DEATH_EVENT=as.factor(df_new2$DEATH_EVENT)

output.forest=ranger(DEATH_EVENT ~ .,data=df_new2, ntree=500,mtry=8)
output.forest
predict(output.forest,testdata[7,-13],type="response")


output.forest2=ranger(DEATH_EVENT ~ .,data=df_new2, ntree=250,mtry=8)
output.forest2

output.forest3=ranger(DEATH_EVENT ~ .,data=df_new2, ntree=650,mtry=8)
output.forest3

output.forest4=ranger(DEATH_EVENT ~ .,data=df_new2, ntree=650,mtry=5)
output.forest4

output.forest5=ranger(DEATH_EVENT ~ .,data=df_new2, ntree=650,mtry=3)
output.forest5
