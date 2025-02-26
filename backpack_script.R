##############################################################################################################
# Script contains eda for the backpack prediction competition for kaggle
##############################################################################################################

library(dplyr)
library(ggplot2)
library(brms)
library(caret)


setwd('D:\\Kaggle\\Backpack Prediction')

train = read.csv('.\\train.csv')
train = train |> mutate(Weight.Capacity..kg. = ifelse(is.na(Weight.Capacity..kg.), 
                                                      mean(Weight.Capacity..kg., na.rm = T), 
                                                      Weight.Capacity..kg.))

train <- train %>%
  mutate(across(where(~is.character(.)),
                ~ifelse(. == "", "Other", .)))

train <- train %>%
  mutate(weight_bins = cut(Weight.Capacity..kg.,
                           breaks = 5,
                           labels = c('w1', 'w2', 'w3', 'w4', 'w5'))
         )

train = train |> mutate(across(where(is.character), as.factor))

#write.csv(train, file = '.\\train_mod.csv')




test = read.csv('.\\test.csv')

test = test |> mutate(Weight.Capacity..kg. = ifelse(is.na(Weight.Capacity..kg.), 
                                                      mean(Weight.Capacity..kg., na.rm = T), 
                                                      Weight.Capacity..kg.))

test <- test %>%
  mutate(across(where(~is.character(.)),
                ~ifelse(. == "", "Other", .)))

test <- test %>%
  mutate(weight_bins = cut(Weight.Capacity..kg.,
                           breaks = 5,
                           labels = c('w1', 'w2', 'w3', 'w4', 'w5'))
  )


test = test |> mutate(across(where(is.character), as.factor))

#write.csv(test, file = '.\\test_mod.csv')




head(train)
str(train)
summary(train)
summary(test)
# Price ------------------------------------------------------------------------------------------------------
# Transformations

hist(scale(train$Price, center = T, scale = F))

hist(-log(1-train$Price))

hist(log(train$Price))

hist(sqrt(train$Price))

hist(1/(train$Price)) # Maybe fit a skewed model for this?

bc = MASS::boxcox(lm(Price~1, data = train))

which.max(bc$y)

bc$x[68]

bct = train$Price^0.7070707-1/0.7070707
hist(bct)


# BootStrapping ----------------------------------------------------------------------------------------------


# We have 7 categorical variables.  Most likely to compare them with price and weight, 

# Function to perform bootstrap resampling
bootstrap_mean <- function(data, n_bootstrap = 1000) {
  res <- data %>%
    group_by(Color) %>%
    summarise(
      mean_bootstrap = list(replicate(n_bootstrap, mean(sample(Price, replace = TRUE)))),
      .groups = "drop"
    )
  
  # Convert bootstrap samples to a data frame
  res %>%
    tidyr::unnest(cols = c(mean_bootstrap)) %>%
    group_by(Color) %>%
    summarise(
      mean_estimate = mean(mean_bootstrap),
      ci_lower = quantile(mean_bootstrap, 0.025),
      ci_upper = quantile(mean_bootstrap, 0.975),
      .groups = "drop"
    )
}

# Run bootstrap
bootstrap_results <- bootstrap_mean(train)
print(bootstrap_results)

ggplot(bootstrap_results, mapping = aes(x = Color, y = mean_estimate))+geom_line() + geom_point()+
  geom_errorbar(aes(ymin = ci_lower, ymax = ci_upper)) +
  labs(title = 'Bootstrap Estimates of the average price for each Bookbag Color', x ='', y = '')

cor(train$Price, train$Weight.Capacity..kg., na.rm = T)


ks.test(train$Price, "punif", min(train$Price), max(train$Price)) # price is uniformly distributed





# Predicting weight capacity to fill the NA values------------------------------------------------------------

hist(train$Weight.Capacity..kg.)
range(train$Weight.Capacity..kg.)

hist(log(train$Weight.Capacity..kg.))

hist(sqrt(train$Weight.Capacity..kg.))

hist(1/(train$Weight.Capacity..kg.)) # Maybe fit a skewed model for this?

lin.mod1 = lm(sqrt(Weight.Capacity..kg.) ~ Brand + Material + Size + Laptop.Compartment + Waterproof + Color + Price, 
              data = train)
summary(lin.mod1)
plot(lin.mod1)


mean((weight_nona$Weight.Capacity..kg. - (predict(lin.mod1)^2))^2)


ggplot(mapping = aes(x = weight_nona$Weight.Capacity..kg., y= predict(lin.mod1))) + geom_point()



gam.mod1 = glm(Weight.Capacity..kg. ~ Brand + Material + Size+ Laptop.Compartment + Waterproof + Price, 
               data = train, family = Gamma(link = 'inverse'))
summary(gam.mod1)

weight_nona = train |> select(Weight.Capacity..kg.) |> filter(!is.na(Weight.Capacity..kg.)) 

mean((weight_nona$Weight.Capacity..kg. - 1/predict(gam.mod1))^2)


gam.mod2 = glm(Weight.Capacity..kg. ~ Brand + Material + Size+ Laptop.Compartment + Waterproof + Price
               , data = train, family = Gamma(link = 'log'))
summary(gam.mod2)

mean((weight_nona$Weight.Capacity..kg. - exp(predict(gam.mod2)))^2)

gam.mod3 = glm(Weight.Capacity..kg. ~ Brand + Material + Size+ Laptop.Compartment + Waterproof + Price,
               data = train, family = Gamma(link = 'identity'))
summary(gam.mod3)

mean((weight_nona$Weight.Capacity..kg. - predict(gam.mod3))^2)

ggplot(mapping = aes(x = weight_nona$Weight.Capacity..kg., y= 1/predict(gam.mod1))) + geom_point()
ggplot(mapping = aes(x = weight_nona$Weight.Capacity..kg., y= exp(predict(gam.mod2)))) + geom_point()
ggplot(mapping = aes(x = weight_nona$Weight.Capacity..kg., y= predict(gam.mod3))) + geom_point()


# Since we have 138 missing values, we will use the mean to fill these in.  


train |> select(is.numeric) |> cor() #no correlated variables

plot(train$Weight.Capacity..kg.,train$Price)



# Brand ------------------------------------------------------------------------------------------------------

ggplot(data = train, mapping = aes(x = Brand)) + geom_bar()



ggplot(data = train, mapping = aes(x = Material)) + geom_bar()


# ML Models --------------------------------------------------------------------------------------------------


tc = trainControl(method = 'cv',
             number = 3)

rfGrid <-  expand.grid(mtry = c(1,2))



set.seed(825)
rf.fit <- train(Price ~ Brand + Material + Size + Laptop.Compartment + Waterproof + Color + weight_bins, 
                data = train, 
                 method = "rf", 
                 trControl = tc,
                 tuneGrid = rfGrid)
rf.fit
plot(varImp(rf.fit))

rf.pred = data.frame('id' = test$id,
                     Price = predict(rf.fit, newdata = test)
                     )


write.csv(rf.pred, file = '.\\rf.pred.csv')


# XBTree -----------------------------------------------------------------------------------------------------


tc = trainControl(method = 'cv',
                  number = 3)

xggrid <-  expand.grid(nrounds =c(100),
                       eta = c(0.1),
                       max_depth = c(2),
                       gamma = c(0),
                       colsample_bytree =c(1),
                       min_child_weight = c(1),
                       subsample = c(1)
)
                       



set.seed(825)
xg.fit <- train(Price ~ Brand + Material + Size + Laptop.Compartment + Waterproof + Color + weight_bins, 
                data = train, 
                method = "xgbTree", 
                trControl = tc,
                tuneGrid = xggrid)
xg.fit
varImp(xg.fit)

xg.pred = data.frame('id' = test$id,
                     Price = predict(xg.fit, newdata = test)
)


write.csv(xg.pred, file = '.\\xg.pred.csv')



# XGBRegression ----------------------------------------------------------------------------------------------

tc = trainControl(method = 'cv',
                  number = 3)

xggrid <-  expand.grid(nrounds =c(100, 200, 300),
                       eta = c(0.1, 0.01, 0.001),
                       lambda = c(0, 1),
                       alpha = c(0, 1)
)




set.seed(825)
xg.fit <- train(Price ~ Brand + Material + Size + Laptop.Compartment + Waterproof + Color + weight_bins, 
                data = train, 
                method = "xgbLinear", 
                trControl = tc,
                tuneGrid = xggrid)
xg.fit
plot(varImp(xg.fit))

xg.pred = data.frame('id' = test$id,
                     Price = predict(xg.fit, newdata = test)
)


write.csv(xg.pred, file = '.\\xg.pred.csv')

# GBM --------------------------------------------------------------------------------------------------------

tc = trainControl(method = 'cv',
                  number = 3)

gbmgrid <-  expand.grid(n.trees =c(200),
                       interaction.depth= c(2),
                       shrinkage = c(0.1),
                       n.minobsinnode = c(1)
)




set.seed(825)
gbm.fit <- train(Price ~ Brand + Material + Size + Laptop.Compartment + Waterproof + Color + weight_bins, 
                data = train, 
                method = "gbm", 
                trControl = tc,
                tuneGrid = gbmgrid)
gbm.fit
plot(varImp(gbm.fit))

gbm.fit$bestTune

gbm.pred = data.frame('id' = test$id,
                     Price = predict(gbm.fit, newdata = test)
)


write.csv(gbm.pred, file = '.\\gbm.pred.csv')

# transformation implementation ----------------------------------------------------------------------------------------

# Transform X to X* in [0,1]
X_star <- (train$Price - 15) / (150 - 15)

hist(X_star)

# Fit logistic regression model
model <- glm(X_star ~ Brand + Material + Size + Laptop.Compartment + Waterproof + Color + weight_bins, 
             data = train,
             family = binomial(link = "probit"))

summary(model)
plot(model)

# Generate new data for prediction
X_star_pred <- predict(model, newdata = train |> select(-Price), type = "response")

# Transform predictions back to original X scale
X_pred <- 15 + (150 - 15) * X_star_pred


plot(train$Price, X_pred)

sqrt(mean((train$Price - X_pred)^2))

X_star_pred <- predict(model, newdata = test, type = "response")

# Transform predictions back to original X scale
X_pred <- 15 + (150 - 15) * X_star_pred


log.pred = data.frame('id' = test$id,
                      Price = X_pred
)


write.csv(log.pred, file = '.\\log.pred.csv')



# Bayesian Approach to the above -----------------------------------------------------------------------------

n = 300000
X_star_brms <- X_star * (n - 1) / n + 1 / (2 * n)

train_brms = train |> mutate(X_star_brms = X_star_brms)



# Fit Bayesian Beta regression using brms
model <- brm(
  bf(X_star_brms ~ Brand + Material + Size + Laptop.Compartment + Waterproof + Color + weight_bins, family = Beta()), # Beta regression
  data = train_brms,
  prior = c(
    prior(normal(0, 2), class = "b"),
    prior(normal(0, 2), class = "Intercept")
  ),
  iter = 2000, warmup = 1000, chains = 1
)

summary(model)

plot(model)



X_star_pred <- predict(model, newdata = test)

# Transform predictions back to original X scale
X_pred <- 15 + (150 - 15) * X_star_pred[,1]

brms.pred = data.frame('id' = test$id,
                      Price = X_pred
)


write.csv(brms.pred, file = '.\\brms.pred.csv')





