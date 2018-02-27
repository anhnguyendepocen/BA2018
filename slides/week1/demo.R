library(GGally)
library(ggplot2)
library(readr)
library(ggbiplot)

records <- read.csv("../../data/trackrecords.txt")
rownames(records) <- records[,1]
records <- records[,-1]
summary(records)


ggscatmat(records)

# Compute PCs using built-in function
z <- prcomp(records, scale=TRUE)

summary(z)
ggscreeplot(z)
ggbiplot(z) + geom_text(label=rownames(records), col='blue',
                        hjust=.5, vjust=1.5, size=3)

#Compute via covariance:
X <- scale(records)
C <- t(X) %*% X
z1 <- eigen(C)
pc.cv <- X %*% z1$vectors

# Compute via svd
z2 <- svd(X)
pc.svd <- X %*% z2$v

qplot(PC1,PC2,data=as.data.frame(z$x))

qplot(PC1,PC2,data=as.data.frame(z$x)) +
  geom_text(label=rownames(records), col='blue',
            hjust=.5, vjust=1.5, size=3)


## EXAMPLE 2
library(ISLR)
summary(USArrests)

ggscatmat(USArrests)
z <- prcomp(USArrests, scale=TRUE)
summary(z)
z
qplot(PC1,PC2, data=as.data.frame(z$x))
qplot(PC1,PC2,data=as.data.frame(z$x)) +
  geom_text(label=rownames(USArrests), col='blue',
            hjust=.5, vjust=1.5, size=3)

ggscreeplot(z)
ggbiplot(z)
