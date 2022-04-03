# setwd('/home/harry/Documents/Oxford/Astrophysics/Projects/MOPED-GP-Expansion/comgp/')
library(lhs)
nlhs = seq(200, 1000, by = 100)
d = 6

for (n in nlhs){
	X = maximinLHS(n, d)

	# filename
	file = paste('lhs/', 'samples_', as.character(n), '.csv', sep ='')

	# write output
	write.csv(X, file)
}
