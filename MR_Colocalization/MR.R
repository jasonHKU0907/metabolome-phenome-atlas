rm(list = ls())     
library("TwoSampleMR")
library("MRInstruments")
library("data.table")


setwd("/public/share/tmp/Meta/MR/")
leadSNP_folder = "/public/home/qiangyx/Metabolomics/MR/leadSNP_for_MR"
R10_manifest <- fread("/public/home/qiangyx/Metabolomics/MR/R10_manifest.csv")
mr_sub1 <- fread("/public/share/tmp/Meta/MR/mr_sub1.csv")
samplesize <- fread("/public/share/tmp/Meta/Target_GWAS/data/samplesize.csv")

for(outcome in unique(mr_sub1$NAME)[1:75]) {
  output_folder = paste0("/public/share/tmp/Meta/MR/result/", outcome)
  outcome_file = paste0("/public/share/tmp/Meta/Target_GWAS/results/geno_assoc_", outcome, ".fastGWA")
  N <- samplesize[samplesize$disease == as.character(outcome),]$N
  if (!dir.exists(output_folder)) {
    dir.create(output_folder, recursive = TRUE)
  }
  ##### Outcome #####
  outcome_data <- read_outcome_data(
    filename = outcome_file,
    sep = "\t",
    snp_col = "SNP",
    beta_col = "BETA",
    se_col = "SE", 
    effect_allele_col = "A1", 
    other_allele_col = "A2", 
    pval_col = "P",
    chr_col = "CHR",
    pos_col = "POS",
    eaf_col = "AF1"
  )
  outcome_data$outcome = outcome
  
  ##### Exposure #####
  for (exposure in mr_sub1$NMR_code[mr_sub1$NAME == outcome]) {
    exposure_name = exposure
    leadsnp_file = file.path(leadSNP_folder, exposure, "exposure_data.txt")
    
    if (!file.exists(leadsnp_file)) {
      next 
    }
    
    full_output_f = file.path(output_folder, exposure_name)
    if (!dir.exists(full_output_f)) {
      dir.create(full_output_f, recursive = TRUE)
    }
    
    # 读取exposure数据
    exposure_data <- read_exposure_data(
      filename = leadsnp_file,
      sep = "\t",
      snp_col = "ID",        
      beta_col = "BETA", 
      se_col = "SE", 
      effect_allele_col = "A1", 
      other_allele_col = "AX", 
      pval_col = "P",
      eaf_col = "A1_FREQ"
    )
    exposure_data$exposure = exposure_name
    write.csv(exposure_data, file = file.path(full_output_f, "exposure_data.csv"), row.names = FALSE)
    dat <- harmonise_data(exposure_dat=exposure_data, outcome_dat=outcome_data) 
    if (nrow(dat) == 1) {
      # 如果 dat 只有一行，使用 Wald ratio
      b_exp <- dat$beta.exposure  
      b_out <- dat$beta.outcome  
      se_exp <- dat$se.exposure  
      se_out <- dat$se.outcome  
      result <- mr_wald_ratio(b_exp = b_exp, b_out = b_out, se_exp = se_exp, se_out = se_out)
      odds_ratios=generate_odds_ratios(result)
      wald_results <- data.frame(
        exposure = dat$exposure[1],
        outcome = dat$outcome[1],
        method = "Wald ratio",
        nsnp = 1,
        b = result$b,
        se = result$se ,
        pval = result$pval,
        or=odds_ratios$or,
        lo_or=odds_ratios$lo_ci,
        up_or=  odds_ratios$up_ci
      )
      write.csv(wald_results, file = file.path(full_output_f, "wald_results.csv"), row.names = FALSE)
      
    } 
    else if (nrow(dat) > 1) {
      
      # if(nrow(dat) <= 3) {
      #next
      #} 
      results <- mr( dat)
      results.withOR <- generate_odds_ratios( results)
      dat$samplesize.outcome<-N 
      dat$samplesize.exposure<-231013 
      dat.mr_heterogeneity<-mr_heterogeneity( dat)   
      dat.mr_pleiotropy_test<-mr_pleiotropy_test( dat) 
      write.csv( dat.mr_heterogeneity,file=file.path(full_output_f,"dat.mr_heterogeneity.csv"),row.names = FALSE)
      write.csv( dat.mr_pleiotropy_test, file=file.path(full_output_f,"dat.mr_pleiotropy_test.csv"),row.names = FALSE)
      leaveoneout <- mr_leaveoneout( dat)
      leaveoneout.withOR <- generate_odds_ratios( leaveoneout)
      write.csv( leaveoneout.withOR, file=file.path(full_output_f,"leaveoneout.withOR.csv"),row.names = FALSE)
      single_snp_analysis <- mr_singlesnp(dat) 
      results.single_snp_analysis.withOR <- generate_odds_ratios(single_snp_analysis)
      write.csv(results.single_snp_analysis.withOR, file=file.path(full_output_f,"single_snp_analysis.withOR.csv"),row.names = FALSE)
      #计算F统计量，评估工具变量的强度
      dat$R<-get_r_from_pn(dat$pval.exposure,dat$samplesize.exposure) 
      dat$"R_2"<-dat$R*dat$R
      dat<-plyr::ddply(dat,"exposure",transform,k=length(exposure)) 
      dat<-plyr::ddply(dat,"exposure",transform,exp.R2=sum(R_2)) 
      dat$"F_stat"<-(dat$exp.R2*(dat$samplesize.exposure-1-dat$k))/((1-dat$exp.R2)*dat$k)
      
      #计算统计效能，大于0.8较好
      #这个参数我不看，样本量一直没改
      calPowerofBinary_1.2 <- function(outcomeCase,outcomeControl) {
        ratio = outcomeControl / outcomeCase
        n = outcomeCase + outcomeControl
        OR = log(1.2)
        sig = 0.05
        rsq = dat$exp.R2
        power = pnorm(sqrt(n*rsq*(ratio/(1+ratio))*(1/(1+ratio)))*OR-qnorm(1-sig/2))
        return(power)
      }
      dat$'power_1.2' = calPowerofBinary_1.2(21982, 41944)
      
      calPowerofBinary_1.45 <- function(outcomeCase,outcomeControl) {
        ratio = outcomeControl / outcomeCase
        n = outcomeCase + outcomeControl
        OR = log(1.45)
        sig = 0.05
        rsq = dat$exp.R2
        power = pnorm(sqrt(n*rsq*(ratio/(1+ratio))*(1/(1+ratio)))*OR-qnorm(1-sig/2))
        return(power)
      }
      dat$'power_1.45' = calPowerofBinary_1.45(21982, 41944)
      
      #整理结果
      results.withOR$egger_intercept[results.withOR$method == 'MR Egger'] = dat.mr_pleiotropy_test$egger_intercept
      results.withOR$MR_Egger.Q[results.withOR$method == 'MR Egger'] = dat.mr_heterogeneity$Q[dat.mr_heterogeneity$method == 'MR Egger']
      results.withOR$MR_Egger.Q_df[results.withOR$method == 'MR Egger'] = dat.mr_heterogeneity$Q_df[dat.mr_heterogeneity$method == 'MR Egger']
      results.withOR$MR_Egger.Q_pval[results.withOR$method == 'MR Egger'] = dat.mr_heterogeneity$Q_pval[dat.mr_heterogeneity$method == 'MR Egger']
      results.withOR$Inverse_variance_weighted.Q[results.withOR$method == 'Inverse variance weighted'] = dat.mr_heterogeneity$Q[dat.mr_heterogeneity$method == 'Inverse variance weighted']
      results.withOR$Inverse_variance_weighted.Q_df[results.withOR$method == 'Inverse variance weighted'] = dat.mr_heterogeneity$Q_df[dat.mr_heterogeneity$method == 'Inverse variance weighted']
      results.withOR$Inverse_variance_weighted.Q_pval[results.withOR$method == 'Inverse variance weighted'] = dat.mr_heterogeneity$Q_pval[dat.mr_heterogeneity$method == 'Inverse variance weighted']
      results.withOR$egger_intercept.pval[results.withOR$method == 'MR Egger'] = dat.mr_pleiotropy_test$pval
      results.withOR$'power_1.2'[results.withOR$method == 'Inverse variance weighted'] = calPowerofBinary_1.2(21982, 41944)
      results.withOR$'power_1.45'[results.withOR$method == 'Inverse variance weighted'] =  calPowerofBinary_1.45(21982, 41944)
      
      write.csv(results.withOR, file=file.path(full_output_f,"results.csv"),row.names = FALSE)
      write.csv(dat, file=file.path(full_output_f,"dat.csv"),row.names = FALSE)
    }
  }
}


