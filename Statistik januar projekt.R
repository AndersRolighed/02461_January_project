setwd("/Users/saraandersen/Desktop/OneDrive/Sara/DTU/02461 Introduktion til intelligente systemer E21/Januar Projekt/clipboard.txt")

# ANOVA dice
BCE_dice<- read.table("BCE_DICE.csv", header = FALSE, sep = ",")
focal_dice <- read.table("focal_dice.csv", header = FALSE, sep = ",")

BCE_dice_unlisted<-unlist(BCE_dice, use.names = FALSE)
focal_dice_unlisted<-unlist(focal_dice, use.names = FALSE)

dice <- c(BCE_dice_unlisted,focal_dice_unlisted)
treatm_dice <- factor(c(rep(c("BCE"), length(BCE_dice_unlisted)),rep(c("focal"), length(focal_dice_unlisted))))

anova(lm(dice~treatm_dice))

# Hvis p-værdien er under vores signifikansniveau (0.05?) er der evidens mod H0 
# (at IOU for BCE og focal loss er ens) og derved må den ene model være bedre end den anden

# Vi kan også bestemme konfidensintervallerne ved,
t.test(BCE_dice_unlisted)
t.test(focal_dice_unlisted)

###############################################################################
# ANOVA IOU
BCE_IOU<- read.table("BCE_IOU.csv", header = FALSE, sep = ",")
focal_IOU <- read.table("focal_iou.csv", header = FALSE, sep = ",")

BCE_IOU_unlisted<-unlist(BCE_IOU, use.names = FALSE)
focal_IOU_unlisted<-unlist(focal_IOU, use.names = FALSE)

IOU <- c(BCE_IOU_unlisted,focal_IOU_unlisted)
treatm_IOU <- factor(c(rep(c("BCE"), length(BCE_IOU_unlisted)),rep(c("focal"), length(focal_IOU_unlisted))))

anova(lm(IOU~treatm_IOU))

# Hvis p-værdien er under vores signifikansniveau (0.05?) er der evidens mod H0 
# (at IOU for BCE og focal loss er ens) og derved må den ene model være bedre end den anden

# Vi kan også bestemme konfidensintervallerne ved,
t.test(BCE_IOU_unlisted)
t.test(focal_IOU_unlisted)

###############################################################################
# ANOVA pixel
BCE_pixel<- read.table("BCE_PIXEL.csv", header = FALSE, sep = ",")
focal_pixel <- read.table("focal_pixel.csv", header = FALSE, sep = ",")

BCE_pixel_unlisted<-unlist(BCE_pixel, use.names = FALSE)
focal_pixel_unlisted<-unlist(focal_pixel, use.names = FALSE)

pixel <- c(BCE_pixel_unlisted,focal_pixel_unlisted)
treatm_pixel <- factor(c(rep(c("BCE"), length(BCE_pixel_unlisted)),rep(c("focal"), length(focal_pixel_unlisted))))

anova(lm(pixel~treatm_pixel))

# Hvis p-værdien er under vores signifikansniveau (0.05?) er der evidens mod H0 
# (at IOU for BCE og focal loss er ens) og derved må den ene model være bedre end den anden

# Vi kan også bestemme konfidensintervallerne ved,
t.test(BCE_pixel_unlisted)
t.test(focal_pixel_unlisted)













