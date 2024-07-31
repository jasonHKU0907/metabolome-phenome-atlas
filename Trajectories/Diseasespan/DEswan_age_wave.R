rm(list=ls())
library(DEswan)
library(data.table)

metabolite_data <- as.data.frame(fread("~/NMR_Preprocessed.csv", sep = ",", header = T))
covariates_df <- as.data.frame(fread("~/Covariates.csv", sep = ",", header = T))
metabolite_data <- merge(covariates_df, metabolite_data, by = "eid")

##DEswan
print(paste0("Processing buckets size:", buckets_size))
res.DEswan=DEswan(data.df = metabolite_data[, -c(1:9)],
                  qt = metabolite_data$Age,
                  window.center = seq(40, 70, by = 1),
                  buckets.size = buckets_size,
                  covariates = metabolite_data[, c(3,4)])
writepath <- paste0("~/deswan_buckets_size_", buckets_size, "_p_df.txt")
fwrite(res.DEswan$p, writepath,sep="\t",row.names=F,col.names=T,quote=F)
writepath <- paste0("~/deswan_buckets_size_", buckets_size, "_coeff_df.txt")
fwrite(res.DEswan$coeff, writepath,sep="\t",row.names=F,col.names=T,quote=F)