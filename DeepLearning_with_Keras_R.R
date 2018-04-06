# Satellite Image Classification using Deep Neural Network with Keras in R with GPU Support (Windows 10)'


#### Import packages
library(rgdal)
library(raster)
library(dplyr)
library(RStoolbox)
library(plyr)
library(keras)
library(tfruns)
library(tfestimators)


#### Setworking directory
setwd("F:\\DeepLearning_tutorial\\Satellite_Image_Calssification\\h20_R_ImageCalssification\\keras_R")

####  Load data 
point<-read.csv("point_data.csv", header=T)
grid<-read.csv("grid_data.csv",header=T)

#### Create a data frame and clean the data
point.df<-cbind(point[c(4:13)],Class_ID=point$Class)
grid.df<-cbind(grid[c(4:13)])
grid.xy<-grid[c(3,1:2)]

#### Convert Class to dummay variables
point.df[,11] <- as.numeric(point.df[,11]) -1

#### Convert data as matrix
point.df<- as.matrix(point.df)
grid.df <- as.matrix(grid.df)

#### Set  `dimnames` to `NULL`
dimnames(point.df) <- NULL
dimnames(grid.df) <- NULL

#### Standardize_the data ((x-mean(x))/sd(x))
point.df[, 1:10] = scale(point.df[, 1:10])
grid.df[, 1:10] = scale(grid.df[, 1:10])

### Split data 
##  Determine sample size
ind <- sample(2, nrow(point.df), replace=TRUE, prob=c(0.80, 0.20))
# Split the `Split data
training <- point.df[ind==1, 1:10]
test <- point.df[ind==2, 1:10]
# Split the class attribute
trainingtarget <- point.df[ind==1, 11]
testtarget <- point.df[ind==2, 11]

#### Hyperparameter flag
FLAGS <- flags(
  flag_numeric('dropout_1', 0.2, 'First dropout'),
  flag_numeric('dropout_2', 0.2, 'Second dropout'),
  flag_numeric('dropout_3', 0.1, 'Third dropout'),
  flag_numeric('dropout_4', 0.1, 'Forth dropout')
)

### Define model parameters with 4 hiden layers with 200 neuron
model <- keras_model_sequential()
model %>% 
  # Imput layer
  layer_dense(units = 200, activation = 'relu', 
              kernel_regularizer =regularizer_l1_l2(l1 = 0.00001, l2 = 0.00001),input_shape = c(10)) %>% 
  layer_dropout(rate = FLAGS$dropout_1,seed = 1) %>% 
  # Hiden layers
  layer_dense(units = 200, activation = 'relu',
              kernel_regularizer = regularizer_l1_l2(l1 = 0.00001, l2 = 0.00001)) %>%
  layer_dropout(rate = FLAGS$dropout_2,seed = 1) %>%
  layer_dense(units = 200, activation = 'relu',
              kernel_regularizer = regularizer_l1_l2(l1 = 0.00001, l2 = 0.00001)) %>%
  layer_dropout(rate = FLAGS$dropout_3,seed = 1) %>%
  layer_dense(units = 200, activation = 'relu',
              kernel_regularizer = regularizer_l1_l2(l1 = 0.0001, l2 = 0.00001)) %>%
  layer_dropout(rate = FLAGS$dropout_4) %>%
  # Output layer
  layer_dense(units = 5, activation = 'softmax')
summary(model)

#### Define an optimizer (Stochastic gradient descent optimizer)
optimizer <- optimizer_sgd(lr = 0.001)

#### Complie the model
model %>% compile(
  loss = 'sparse_categorical_crossentropy',
  optimizer = optimizer,
  metrics = 'accuracy'
)

####  Fit the model to the data 
history<-model %>% fit(
  training, trainingtarget, 
  epochs = 100, 
  batch_size = 100, 
  shuffle = TRUE,
  validation_split = 0.2,
  callbacks = callback_tensorboard()
)

### Plot history

plot(history)

#### Evaluate the model
score <- model %>% evaluate(test, testtarget, batch_size = 100)
cat('Test loss:', score[[1]], '\n')
cat('Test accuracy:', score[[2]], '\n')

#### Prediction & confusion matrix - test data
class.test <- model %>%
  predict_classes(test, batch_size = 100)
table(testtarget,class.test)

#### Predicted Class Probability

prob.test <- model %>%
  predict_proba(test, batch_size = 100)

#### Prediction at grid data
Class.grid <- model %>%
  predict_classes(grid.df, batch_size = 100)

#### Detach keras, tfruns, tftestimators

detach(package:keras, unload=TRUE)
detach(package:tfruns, unload=TRUE)
detach(package:tfestimators, unload=TRUE)

#### Change column name
class<-as.data.frame(Class.grid)
new.grid<-cbind(x=grid.xy$x, y=grid.xy$y,Class_ID=class )
names(new.grid)
colnames(new.grid)[3]<-"Class_ID"
new.grid.na<-na.omit(new.grid)

#### Load ID file

#### Join Class Id Column
ID<-read.csv("Landuse_ID.csv", header=TRUE)
ID

#### Convert raster

#### Convert to raster
x<-SpatialPointsDataFrame(as.data.frame(new.grid.na)[, c("x", "y")], data = new.grid.na)
r <- rasterFromXYZ(as.data.frame(x)[, c("x", "y", "Class_ID")])

myPalette <- colorRampPalette(c("darkgoldenrod1","red", "darkgreen","green", "blue"))
spplot(r,"Class_ID",  
       colorkey = list(space="right",tick.number=1,height=1, width=1.5,
                       labels = list(at = seq(0,3.8,length=5),cex=1.0,
                                     lab = c("Class-1 (Road/parking/pavement)" ,"Class-2 (Building)", "Class-3 (Tree/buses)", "Class-4 (Grass)", "Class-5 (Water)"))),
       col.regions=myPalette,cut=4)
writeRaster(r,"predicted_Landuse.tiff","GTiff",overwrite=TRUE)

#### Clean everyrhing

gc()
