rm(list=ls())
library(data.table)
library(dplyr)
library(MatchIt)

match_control <- function(data) {
    m.out <- matchit(target_y ~ Sex + Race + Age + BMI + TDI, data = data,
                     method = "nearest", distance = "glm" , ratio = 5, link = "logit",
                     exact = ~ Sex + Race, caliper = c(.2, Age = 3, BMI = 5),
                     std.caliper = c(TRUE, FALSE, FALSE))
    m.data <- match.data(m.out)

    return(m.data)
}
for (i in 1:length(disease_list)) {
  disease <- disease_list[i]
  data <- disease_target_data[disease_target_data$target_disease == disease, ]
  data <- left_join(data, covariates_df, by = "eid")
  matched_result <- match_control(data)
  writepath <- paste0("path_to_matched_result/", disease, "_matched_result.txt")
  fwrite(matched_result, writepath, sep = "\t", row.names = F, col.names = T, quote = F)
  print(i)
}


rm(list=ls())
library(data.table)
library(dplyr)
covariates_df <- as.data.frame(fread("../Covariates.csv", sep = ",", header = T))
metabolite_data <- merge(covariates_df, metabolite_data, by = "eid")

for (metabolite in metabolite_list) {
  formula <- as.formula(paste0(metabolite, " ~ Smoke + Statin + FastingTime"))
  metabolite_data[which(!is.na(metabolite_data[, metabolite])), metabolite] <- resid(lm(formula, data = metabolite_data))
}
metabolite_data <- metabolite_data[ , -c(2,3,4)]

result <- data.frame(Metabolite = metabolite_list)
for (bin in bins_level) {
  result[[bin]] <- NA
}
for (i in 1:length(disease_list)) {
  disease <- disease_list[i]
  disease_result <- result
  matched_data <- fread(paste0("path_to_matched_result/", disease, "_matched_result.txt"), sep = "\t", header = T)
  matched_data <- matched_data%>%
    group_by(subclass) %>%
    mutate(BL2Target_yrs_for_1 = ifelse(target_y == 1, BL2Target_yrs, 0)) %>%
    mutate(BL2Target_yrs = sum(BL2Target_yrs_for_1)) %>%
    ungroup() %>%
    select(-BL2Target_yrs_for_1) %>%
    bind_rows()
  matched_data <- as.data.frame(matched_data)
  
  matched_data <- matched_data[matched_data$BL2Target_yrs <=15 & matched_data$BL2Target_yrs >= 0, ]
  matched_data$BL2Target_yrs <- -matched_data$BL2Target_yrs
  matched_data$bin <- cut(matched_data$BL2Target_yrs, bins)
  
  for (bin in bins_level) {
    data <- matched_data[matched_data$bin == bin, ]
    case_ids <- data[data$target_y == 1,]$eid
    control_ids <- data[data$target_y == 0,]$eid
    case_metabolite_data <- metabolite_data[metabolite_data$eid %in% case_ids,][,-1]
    case_metabolite_means <- colMeans(case_metabolite_data, na.rm = TRUE)
    control_metabolite_data <- metabolite_data[metabolite_data$eid %in% control_ids,][,-1]
    control_metabolite_means <- colMeans(control_metabolite_data, na.rm = TRUE)
    control_metabolite_sd <- apply(control_metabolite_data, 2, sd, na.rm = TRUE)
    case_control_meta_zscore <- (case_metabolite_means-control_metabolite_means)/control_metabolite_sd
    disease_result[[bin]] <- case_control_meta_zscore
  }
  writepath <- paste0("path_to_matrix_result/", disease, "_metabolite_matrix_result.txt")
  fwrite(disease_result, writepath, sep = " ", row.names = F, col.names = T, quote = F, na = "NA")
  print(i)
}