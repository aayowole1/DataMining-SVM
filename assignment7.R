# 4/18/23

# Importing the dataset
dataset = read.csv('Iris_Data.csv')

# Encoding categorical data
dataset$Species = factor(dataset$Species,
                         levels = c('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'),
                         labels = c(1, 2, 3))

# Feature Scaling
scaled_cols = scale(dataset[, 1:2])
dataset[, 1:2] = scaled_cols

# Fitting SVM to the Training set
library(e1071)
classifier = svm(formula = Species ~ .,
                 data = dataset,
                 type = 'C-classification',
                 kernel = 'radial')

# Predicting the Test set results
y_pred = predict(classifier, newdata = dataset)

# Showing the Confusion Matrix and Accuracy
library(caret)
cm = confusionMatrix(y_pred, dataset$Species)
print(cm$table)
print(cm$overall['Accuracy'])

# Visualizing the  dataset results
set = dataset
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Sepal.Length', 'Sepal.Width')
y_grid = predict(classifier, newdata = grid_set)
plot(NULL,
     main = 'SVM (Training set)',
     xlab = 'Sepal.Width (Scaled)', ylab = 'Sepal.Length (Scaled)',
     xlim = range(X1), ylim = range(X2))
points(grid_set, pch = 20, col = c('tomato', 'springgreen3', 'gold')[y_grid])
points(set, pch = 21, bg = c('red3', 'green4', 'yellow')[set$Species])
