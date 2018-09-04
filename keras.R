rm(list = ls())
.rs.restartR()
library(tidyverse)
library(keras)
use_condaenv("r-tensorflow") 

n_epochs <- 20

mnist <- dataset_mnist()

x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y

# reshape
dim(x_train) <- c(nrow(x_train), 784)
dim(x_test) <- c(nrow(x_test), 784)


# rescale
x_train <- x_train / 255
x_test <- x_test / 255

y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)


# Specify models
model_list <- list()

# Q1
model_list[["Dropout"]] <- keras_model_sequential() %>%
  layer_dense(units = 256, activation = "relu", input_shape = c(784)) %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 10, activation = "softmax") %>% 
  compile(loss = "categorical_crossentropy",
          optimizer = optimizer_rmsprop(),
          metrics = c("accuracy"))

# Q2 
model_list[["Extra layer"]] <- keras_model_sequential() %>%
  layer_dense(units = 256, activation = "relu", input_shape = c(784)) %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 10, activation = "softmax") %>% 
  compile(loss = "categorical_crossentropy",
          optimizer = optimizer_rmsprop(),
          metrics = c("accuracy"))

# Q3
model_list[["Early stopping"]] <- model_list[["Dropout"]]

# Q4
model_list[["Regularisation"]] <- keras_model_sequential() %>%
  layer_dense(units = 256, activation = "relu", input_shape = c(784),
              kernel_regularizer = regularizer_l1(1e-3)) %>%
  layer_dense(units = 128, activation = "relu",
              kernel_regularizer = regularizer_l1(1e-3)) %>%
  layer_dense(units = 10, activation = "softmax") %>% 
  compile(loss = "categorical_crossentropy",
          optimizer = optimizer_rmsprop(),
          metrics = c("accuracy"))


# Fit models
fit_list <- list()

# Q1
fit_list[["Dropout"]] <- model_list[["Dropout"]] %>% 
  fit(x_train, y_train, epochs = n_epochs, batch_size = 128,
      validation_split = 0.2)

# Q2 
fit_list[["Extra layer"]] <- model_list[["Extra layer"]] %>% 
  fit(x_train, y_train, epochs = n_epochs, batch_size = 128,
      validation_split = 0.2)

# Q3
fit_list[["Early stopping"]] <- model_list[["Early stopping"]] %>% 
  fit(x_train, y_train, epochs = n_epochs, batch_size = 128, 
      validation_split = 0.2,
      callbacks = callback_early_stopping(monitor = 'val_loss', 
                                          patience = 2))

# Q4
fit_list[["Regularisation"]] <- model_list[["Regularisation"]] %>% 
  fit(x_train, y_train, epochs = n_epochs, batch_size = 128,
      validation_split = 0.2)


# Evaluate models
evaluate_df <- function(x) {
  evaluate(x, x_test, y_test,verbose = 0) %>% 
    bind_rows()
}

model_df <- data_frame(
  name = names(model_list),
  model = model_list,
  fit = fit_list
) %>% 
  mutate(eval = map(model, evaluate_df)) %>% 
  unnest(eval)

k_clear_session()

# Plots
walk(model_df$fit, plot)

model_df %>% 
  select(name, loss, acc) %>% 
  gather(var, val, -name) %>% 
  ggplot(aes(x = name, y = val)) + 
  geom_col() +
  facet_wrap(~var) +
  coord_flip()