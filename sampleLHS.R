# setwd('/home/harry/Documents/Oxford/Astrophysics/Projects/MOPED-GP-Expansion/comgp/')
library(lhs)
n = 20
d = 6
X = maximinLHS(n, d)
write.csv(X, 'lhs/samples_20.csv')
