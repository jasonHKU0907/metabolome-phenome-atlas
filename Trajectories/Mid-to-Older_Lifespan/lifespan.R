rm(list=ls())
library(data.table)
library(dplyr)
library(ggplot2)
library(broom)
library(showtext)
font_add("Arial", "~/Arial.ttf")
showtext_auto()

metabolite_data <- merge(covariates_df, metabolite_data, by = "eid")

#Adjust for Sex, Race, TDI, BMI, Smoke, Statin, and Fasting time
for (metabolite in metabolite_list) {
  formula <- as.formula(paste0(metabolite, " ~ Sex + Race + TDI + BMI + Smoke + Statin + FastingTime"))
  metabolite_data[which(!is.na(metabolite_data[, metabolite])), metabolite] <- resid(lm(formula, data = metabolite_data))
}
metabolite_data <- metabolite_data[ , -c(3:9)]

plot_df <- data.frame()
for (metabolite in metabolite_list) {
  data <- metabolite_data[ , c("Age", metabolite)]
  data <- na.omit(data)
  formula <- as.formula(paste0(metabolite, " ~ Age"))
  model <- loess(formula, data, span = 0.8)
  predicted_zscores <- predict(model, newdata = data.frame(Age = age_year))
  new_data <- data.frame(Metabolite = metabolite, Age = age_year, zscore = predicted_zscores)
  plot_df <- bind_rows(plot_df, new_data)
  print(metabolite)
}
fwrite(plot_df, "loess_imputed_aging_trajectoies.txt",sep=",",row.names=F,col.names=T,quote=F)


##sex age interaction
rm(list=ls())
library(data.table)
library(dplyr)
library(broom)
metabolite_data <- merge(covariates_df, metabolite_data, by = "eid")
metabolite_data$Sex <- factor(metabolite_data$Sex)
metabolite_data$Race <- factor(metabolite_data$Race)
output <- data.frame()
for (metabolite in metabolite_list) {
  formula <- as.formula(paste0(metabolite, " ~ Age + Sex + Age*Sex + Race + TDI + BMI + Smoke + Statin + FastingTime"))
  model <- lm(formula, data = metabolite_data)
  result <- as.data.frame(tidy(model))
  result <- result[c(2, 10), c(2, 5)]
  result <- t(data.frame(unlist(as.vector(result))))
  result <- cbind(Metabolite = metabolite, result)
  output <- rbind(output, result)
}
colnames(output) <- c("Metabolite", "Estimate_Age", "Estimate_Age:Sex", "p_Age", "p_Age:Sex")
output$p_Age <- as.numeric(output$p_Age)
output$`p_Age:Sex` <- as.numeric(output$`p_Age:Sex`)
output$p_bonf_Age <- output$p_Age * 313
output$`p_bonf_Age:Sex` <- output$`p_Age:Sex` * 313
writepath <- paste0("age_sex_interaction.csv")
fwrite(output, writepath, sep=",", row.names=F, col.names=T, quote=F)