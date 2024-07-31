library(data.table)
library("coloc")
library(dplyr)
setwd("/public/share/tmp/Meta/coloc/")
p1 <- 1e-4
p2 <- 1e-4
p12 <- 1e-5
window <- 1e6
sub1 <- fread("sub1.csv")
sub1=sub1[1:47,]
samplesize <- fread("/public/share/tmp/Meta/Target_GWAS/data/samplesize.csv")
all_pheno_leadsnp_new <- fread("/public/home/qiangyx/Metabolomics/all_pheno_leadsnp_new.tsv")
outcome_names <- unique(sub1$disease)

for(outcome_name in outcome_names) {
  result_dir <- sprintf("/public/share/tmp/Meta/coloc/result/%s", outcome_name)
  if (!dir.exists(result_dir)) {
    dir.create(result_dir, recursive = TRUE)
  }
  
  
  if (outcome_name %in% samplesize$disease) {
    outcome_data <- samplesize[disease == outcome_name]
    Nsum <- outcome_data$N
  } else {
    print(paste("未找到对应的phenocode:", phenocode, "在manifest文件中。")) 
    Nsum <- NA 
  }
  
  outcome_gwas_path <- paste0("/public/share/tmp/Meta/Target_GWAS/results/", "geno_assoc_",outcome_name, ".fastGWA")
  outcome_gwas_data <- fread(outcome_gwas_path)
  colnames(outcome_gwas_data)[1] <- "chr"
  

  sub1_sub <- subset(sub1,sub1$disease == outcome_name)
  pheno_list <- sub1_sub$ID
  
  for (pheno in pheno_list) {
    dir_path <- paste0("/public/home/qiangyx/Metabolomics/GWAS/UKB_GWAS_new/", pheno, "/")
    pheno_gwas_data <- data.frame()
    ##提取代谢物gwas
    for (chr in 1:22) {
      file_name <- paste0("chr", chr, ".", pheno, ".glm.linear")
      file_path <- paste0(dir_path, file_name)
      if (file.exists(file_path)) {
        temp_data <- fread(file_path)
        pheno_gwas_data <- rbind(pheno_gwas_data, temp_data)
      }
    }
    colnames(pheno_gwas_data)[1] <- "chr"
    ##得到lead snps
    lead_snvs_data=subset(all_pheno_leadsnp_new,all_pheno_leadsnp_new$Pheno==pheno)
    filter_lead_snvs <- function(lead_snvs_data, window){
      lead_snvs_data <- lead_snvs_data[order(lead_snvs_data$P),]
      filtered_lead_snvs <- data.frame()
      for (j in 1:nrow(lead_snvs_data)) {
        current_snv <- lead_snvs_data[j,]
        if (nrow(filtered_lead_snvs) == 0) {
          filtered_lead_snvs <- rbind(filtered_lead_snvs, current_snv)
          next
        }
        is_far_enough <- all(abs(current_snv$POS - filtered_lead_snvs$POS) > window | current_snv$CHR != filtered_lead_snvs$CHR)
        if (is_far_enough) {
          filtered_lead_snvs <- rbind(filtered_lead_snvs, current_snv)
        }
      }
      return(filtered_lead_snvs)
    }
    filtered_lead_snvs <- filter_lead_snvs(lead_snvs_data, window) 
    # 对于每个lead SNP，进行colocalization分析
    snv_list <- as.list(filtered_lead_snvs$ID)
    #定义函数
    run_coloc_analysis <- function(pheno_gwas_data, outcome_gwas_data, snv_list, p1, p2, p12) { 
      coloc_results <- list()
      for (snv in snv_list) {
        lead_snv_position <-  filtered_lead_snvs[filtered_lead_snvs$ID==snv,]$POS
        lead_snv_chromosome <-  filtered_lead_snvs[filtered_lead_snvs$ID==snv,]$CHR
        #1mb的snp 要不要限制基因？
        pheno_sub <- pheno_gwas_data[pheno_gwas_data$chr == lead_snv_chromosome & 
                                       (pheno_gwas_data$POS >= (lead_snv_position - 500000)) & 
                                       (pheno_gwas_data$POS <= (lead_snv_position + 500000))]
        outcome_sub <- outcome_gwas_data[outcome_gwas_data$chr == lead_snv_chromosome & 
                                           (outcome_gwas_data$POS >= (lead_snv_position - 500000)) & 
                                           (outcome_gwas_data$POS <= (lead_snv_position + 500000))] 
        common_snps <- intersect(pheno_sub$ID, outcome_sub$SNP)
        if (length(common_snps) == 0) {
          next
        }
        pheno_snvs_subset_clean <- pheno_sub %>%
          distinct(ID, .keep_all = TRUE) %>%
          filter(ID %in% common_snps) %>%
          arrange(ID)
        outcome_snvs_subset_clean <- outcome_sub %>%
          distinct(SNP, .keep_all = TRUE) %>%
          filter(SNP %in% common_snps) %>%
          arrange(SNP)
        ###coloc.susie?
        coloc_result <- coloc.abf(dataset1=list(
          snp=pheno_snvs_subset_clean$ID,
          beta = pheno_snvs_subset_clean$BETA,
          varbeta = pheno_snvs_subset_clean$SE^2,
          p=pheno_snvs_subset_clean$P, 
          MAF=pheno_snvs_subset_clean$A1_FREQ,
          N= 231013,
          type="quant" ),
          dataset2=list(
            snp=outcome_snvs_subset_clean$SNP,
            beta=outcome_snvs_subset_clean$BETA,
            varbeta=outcome_snvs_subset_clean$SE^2,
            MAF=outcome_snvs_subset_clean$AF1,
            N= Nsum,
            type="cc" ),
          p1 = p1, p2 = p2, p12 = p12)
        
        coloc_results[[snv]] <- coloc_result
      }
      return(coloc_results)
    }
    coloc_results <- run_coloc_analysis(pheno_gwas_data, outcome_gwas_data, snv_list, p1, p2, p12)
    
    # 整理结果
    pp_threshold <- 0.8
    coloc_results_filtered <- data.frame()
    for (snp_name in names(coloc_results)) {
      results_df <- coloc_results[[snp_name]]$results
      snp_filtered <- data.frame(sig_snps = results_df$snp, H4 = results_df$SNP.PP.H4)
      snp_filtered$filtered_lead_snvs <- snp_name
      coloc_results_filtered <- rbind(coloc_results_filtered, snp_filtered)
    }
    
    coloc_results_filtered <- coloc_results_filtered[coloc_results_filtered[["H4"]] > pp_threshold, ]
    rownames(coloc_results_filtered) <- NULL
    if (nrow(coloc_results_filtered) > 0) {
      coloc_results_filtered$Pheno <- pheno
      result_path <- sprintf("%s/%s.csv", result_dir, pheno)
      write.csv(coloc_results_filtered, file = result_path, row.names = FALSE)
    }
  }
  files <- list.files(result_dir, full.names = TRUE, pattern = "\\.csv$")
  all_results <- rbindlist(lapply(files, fread))
  final_result_path <- sprintf("/public/share/tmp/Meta/coloc/result/%s.csv", outcome_name)
  fwrite(all_results, final_result_path)
}
