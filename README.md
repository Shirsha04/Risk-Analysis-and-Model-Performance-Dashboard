# Risk-Analysis-and-Model-Performance-Dashboard
# Load the required libraries
library(shiny)
library(shinydashboard)
library(tidyverse)
library(caret)
library(e1071)
library(pROC)
library(ggplot2)

# Load Data
data <- read.csv(file.choose())  # Choose your dataset file
str(data)

# Preprocess Data
data$State <- as.factor(data$State)
data$City.Or.County <- as.factor(data$City.Or.County)
data$Coordinates_Found <- as.factor(data$Coordinates_Found)

# Create a Risk Score
data$Risk_Score <- with(data, Victims.Killed * 2 + Victims.Injured + Suspects.Killed * 2)

# Normalize Latitude and Longitude
data$Latitude <- scale(data$Latitude)
data$Longitude <- scale(data$Longitude)

# Split Data
set.seed(123)
trainIndex <- createDataPartition(data$Risk_Score, p = 0.7, list = FALSE)
trainData <- data[trainIndex, ]
testData <- data[-trainIndex, ]

# Convert Risk_Score to Factor
trainData$Risk_Score <- as.factor(trainData$Risk_Score)
testData$Risk_Score <- as.factor(testData$Risk_Score)

# Train Naive Bayes Model
nb_model <- naiveBayes(Risk_Score ~ State + Victims.Killed + Victims.Injured, data = trainData)

# Predict on Test Data
predictions_nb <- predict(nb_model, testData)

# Ensure Levels Match Between Predictions and Test Data
predictions_nb <- factor(predictions_nb, levels = levels(testData$Risk_Score))

# Confusion Matrix
cm <- confusionMatrix(predictions_nb, testData$Risk_Score)

# Visualization: Confusion Matrix Plot
cm_df <- as.data.frame(cm$table)
colnames(cm_df) <- c("Reference", "Prediction", "Frequency")

# Define UI for the Dashboard
ui <- dashboardPage(
  dashboardHeader(title = "Model Analysis Dashboard"),
  dashboardSidebar(
    sidebarMenu(
      menuItem("Model Performance", tabName = "model_performance", icon = icon("bar-chart")),
      menuItem("Clustering", tabName = "clustering", icon = icon("cluster"))
    )
  ),
  dashboardBody(
    tabItems(
      # Tab for Model Performance
      tabItem(
        tabName = "model_performance",
        fluidRow(
          box(title = "Confusion Matrix", width = 12, solidHeader = TRUE, status = "primary",
              plotOutput("cm_plot"))
        ),
        fluidRow(
          box(title = "ROC Curve", width = 12, solidHeader = TRUE, status = "primary",
              plotOutput("roc_plot"))
        )
      ),
      # Tab for Clustering
      tabItem(
        tabName = "clustering",
        fluidRow(
          box(title = "K-Means Clustering", width = 6, solidHeader = TRUE, status = "primary",
              plotOutput("kmeans_plot"))
        ),
        fluidRow(
          box(title = "Hierarchical Clustering", width = 6, solidHeader = TRUE, status = "primary",
              plotOutput("hc_plot"))
        )
      )
    )
  )
)

# Define Server Logic
server <- function(input, output) {
  
  # Plot Confusion Matrix
  output$cm_plot <- renderPlot({
    ggplot(data = cm_df, aes(x = Reference, y = Prediction, fill = Frequency)) +
      geom_tile() +
      geom_text(aes(label = Frequency), color = "white") +
      scale_fill_gradient(low = "blue", high = "red") +
      labs(title = "Confusion Matrix", x = "Actual", y = "Predicted") +
      theme_minimal()
  })
  
  # Plot ROC Curve
  output$roc_plot <- renderPlot({
    testData$Risk_Score_numeric <- as.numeric(as.character(testData$Risk_Score))
    predictions_nb_numeric <- as.numeric(as.character(predictions_nb))
    roc_curve <- roc(testData$Risk_Score_numeric, predictions_nb_numeric)
    plot(roc_curve, col = "blue", lwd = 2, main = "ROC Curve for Naive Bayes")
    auc_value <- auc(roc_curve)
    print(paste("AUC:", auc_value))
  })
  
  # K-Means Clustering Plot
  output$kmeans_plot <- renderPlot({
    set.seed(123)
    data_numeric <- data[, sapply(data, is.numeric)]  # Select only numeric columns
    data_normalized <- scale(data_numeric)  # Normalize data
    kmeans_model <- kmeans(data_normalized, centers = 3, nstart = 25)
    data$Cluster <- as.factor(kmeans_model$cluster)
    ggplot(data, aes(x = data_numeric[,1], y = data_numeric[,2], color = Cluster)) +
      geom_point(size = 3) +
      labs(title = "K-Means Clustering Results", x = "Feature 1", y = "Feature 2") +
      theme_minimal()
  })
  
  # Hierarchical Clustering Plot
  output$hc_plot <- renderPlot({
    data_numeric <- data[, sapply(data, is.numeric)]  # Select numeric columns
    data_normalized <- scale(data_numeric)  # Normalize data
    distance_matrix <- dist(data_normalized, method = "euclidean")
    hc_model <- hclust(distance_matrix, method = "complete")
    plot(hc_model, main = "Hierarchical Clustering Dendrogram", xlab = "Samples", ylab = "Height", sub = "", cex = 0.7)
  })
}

# Run the Application
shinyApp(ui = ui, server = server)
