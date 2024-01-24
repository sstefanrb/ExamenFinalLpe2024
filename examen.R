# Parte 1

# kaggle datasets list -s sukuzhanay

# kaggle datasets download -d sukuzhanay/credit-card-fraud

unzip("credit-card-fraud.zip")

# Parte 2 y 3

library(dplyr)
library(janitor)
library(caret)

data = read.csv("dataset_cards.csv")

# EDA

count_type <- data %>%
  group_by(type, isFraud) %>%
  summarise(count = n())

count_fraud <- data %>%
  summarise(total_count = n(), fraud_count = sum(isFraud == 1))

print("Resumen por tipo y isFraud:")
print(count_type)

print("Total de isFraud:")
print(count_fraud)

# LIMPIEZA

data$isFraud <- as.factor(data$isFraud)

data <- data %>% janitor::clean_names()

filtered_data <- data %>% filter(type %in% c("CASH_OUT", "TRANSFER"))

filtered_data$is_fraud <- as.factor(filtered_data$is_fraud)

undersampled_data <- downSample(x = filtered_data[, -11], y = filtered_data$is_fraud)

data_balanced <- data.frame(undersampled_data)

summary(data_balanced$isFraud)

type_encoding <- c("TRANSFER" = 0, "CASH_OUT" = 1)
data_balanced$type_numeric <- as.numeric(factor(data_balanced$type, levels = names(type_encoding), labels = type_encoding))

write.csv(data_balanced, file = "dataset_cleaned.csv", row.names = FALSE)


# Ejercicio 4

library(xgboost)
library(caret)
library(pROC)

# Cargado datos -----------------------------------------------------------

clean = read.csv("dataset_cleaned.csv")

View(clean)

# Preparacion datos -------------------------------------------------------

selected_columns <- c("step", "amount", "oldbalance_org", "newbalance_orig", "oldbalance_dest", "newbalance_dest", "is_fraud", "type_numeric")
selected_dataset <- clean[selected_columns]

set.seed(123)
split_index <- createDataPartition(selected_dataset$is_fraud, p = 0.8, list = FALSE)
train_data <- selected_dataset[split_index, ]
test_data <- selected_dataset[-split_index, ]


matriz_entrenamiento <- xgb.DMatrix(as.matrix(train_data[, -c(7)]), label = train_data$is_fraud)
matriz_prueba <- xgb.DMatrix(as.matrix(test_data[, -c(7)]), label = test_data$is_fraud)


# Ejercicio 5


# Modelo XGBOOST ----------------------------------------------------------

#XGBoost es altamente eficiente y preciso, ofreciendo un rendimiento destacado en una amplia variedad de tareas de aprendizaje automático, 
#incluso con grandes volúmenes de datos. Incorpora técnicas avanzadas como regularización y manejo de valores faltantes, 
#lo que ayuda a mejorar la precisión y prevenir el sobreajuste.

parametros <- list(
  objective = "binary:logistic",
  eval_metric = "logloss"
)

# Entrenamiento

modelo_xgboost <- xgboost(data = matriz_entrenamiento, params = parametros, nrounds = 100, verbose = 1)


# Predicciones

predicciones <- predict(modelo_xgboost, matriz_prueba)


# Normalizacion de predicciones

predicciones_clases <- ifelse(predicciones > 0.5, 1, 0)

# Ejercicio 6

# Resultados

resultado <- table(predicciones_clases, test_data$is_fraud)
print(resultado)


# Curva ROC

curva_roc <- roc(test_data$is_fraud, predicciones)
plot(curva_roc, col = "red", main = "Curva ROC")
auc(curva_roc)

#Final