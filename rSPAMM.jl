## Julia code to reproduce results in rSPAMM


## Init process and define some functions
#region

# Change working directory to the location of current file
@__FILE__
cd(@__DIR__)
pwd()

using Distributions						# to load probability distributions
using Optim								# for optimization
using RData								# to read data from R
using ForwardDiff, PreallocationTools   # packages to calculate the autodiff


# Logit transform
function logitt(x)
	return log(x/(1-x))
end

# Inversed logit transform
function ilogit(x)
	return (1.0)/((1.0)+exp(-x))
end

# posfun function
function posfun(x, eps)
	if x >= eps
		return x
	else
		y = 1.0-x/eps
		return eps/(1+y+y*y)
	end
end

#endregion

## Load data
#region

# Load the dataset harpeast from R
using RData
objs = load("harpeast.rds")
data_harpeast::DictoVec{Any} = objs["data"]
param::DictoVec{Float64} = objs["parameters"]
logK::Float64, Mtilde::Float64, M0tilde::Float64 = param["logK"], param["Mtilde"], param["M0tilde"]

# Define a population data stucture
struct PopData
	Amax::Int								# Maximum age group
	Cdata::Matrix{Int}						# Catch data
	Nc::Int									# Number of years with catch data
	pupProductionData::Matrix{Float64}		# Pup production estimates
	Np::Int									# Number of years with pup production data
	Ftmp::Vector{Float64}					# Observed fecundity rates
	Pmat::Matrix{Float64}					# Birth ogives (empirical cumulative distribution function)	
	Npred::Int								# Number of years to run projections
	priors::Matrix{Float64}					# Priors for parameters
	Npriors::Int							# Number of priors
	CQuota::Vector{Float64}					# Catch level in future projections
end

# [https://discourse.julialang.org/t/passing-struct-vs-struct-fields-as-function-arguments/67584]
# Make a structure of data to pass to function
D::PopData = PopData(convert(Int, data_harpeast["Amax"]), 
			data_harpeast["Cdata"],
			data_harpeast["Nc"],
			data_harpeast["pupProductionData"],
			data_harpeast["Np"],
			data_harpeast["Ftmp"],
			data_harpeast["Pmat"],
			convert(Int,data_harpeast["Npred"]),
			data_harpeast["priors"],
			data_harpeast["Npriors"],
			data_harpeast["CQuota"]
			);
#endregion

## Find maximum likelihood using Optim.jl
#region

# Negative Likelihood function to minimize
function objective_function(x; D)

	Amax = D.Amax							# Maximum age group
	Cdata = D.Cdata							# Catch data
	Nc = D.Nc								# Number of years with catch data
	pupProductionData = D.pupProductionData	# Pup production estimates
	Np = D.Np								# Number of years with pup production data
	Ftmp = D.Ftmp							# Observed fecundity rates
	Pmat = D.Pmat							# Birth ogives (empirical cumulative distribution function)
	Npred = D.Npred							# Number of years to run projections
	priors = D.priors						# Priors for parameters
	Npriors = D.Npriors						# Number of priors
	CQuota = D.CQuota						# Catch level in future projections
			
	logK = x[1]								# Initial population size
	Mtilde = x[2]							# Natural adult mortality
	M0tilde = x[3]							# Natural pup mortality
	
	# Transform estimated parameters
	K = exp(logK)
	M = ilogit(Mtilde)
	M0 = ilogit(M0tilde)

	#Adult and pup survival
	em = exp(-M)
	em0 = exp(-M0)

	Catch = zeros(Nc+Npred+2,3)						# Catch data
	P = zeros(Nc+Npred+1,Amax)						# Birth ogive
	mub = zeros(Npriors)							# Mean value of priors
	sdb = zeros(Npriors)							# SD of priors

	# To make the function accept autodiff
	# [https://discourse.julialang.org/t/how-to-make-this-function-compatible-with-forwarddiff/67415/15]
	N_ = zeros(Nc+Npred+2,Amax)						# Population matrix
	N0_ = zeros(Nc+Npred+2)							# Pup abundance
	N1_ = zeros(Nc+Npred+2)							# Abundance of one year and older seals
	NTot_ = zeros(Nc+Npred+2)						# Total population
	N = get_tmp(dualcache(N_), M)
	N0 = get_tmp(dualcache(N0_), M)
	N1 = get_tmp(dualcache(N1_), M)
	NTot = get_tmp(dualcache(NTot_), M)
	N .= N_
	N0 .= N0_
	N1 .= N1_
	NTot .= NTot_

	Ft = zeros(Nc+Npred+2)							# Fecundity rates
	b = zeros(Npriors)								# Concatenation of parameters
	
	# Initialize values for parameters
	D1 = 0.0
	DNmax = 0.0
	N0CurrentYear = 0.0
	N1CurrentYear = 0.0
	NTotCurrentYear = 0.0
	NTotmax = 0.0
	NTotPred = 0.0

	# Preliminary calculations - Preparations of Catch data and fecundity rates
	Catch[1,:] = [1945, 0, 0]
	Catch[2:Nc+1,:] = Cdata[1:Nc,:] 			# Year, Pup catch, 1+ catch
	Ft[2:Nc+1] = Ftmp[1:Nc]

	for i = Nc+2:Nc+Npred+2
		Catch[i,1] = Catch[i-1,1]+1 			# Year
	end
	Catch[(Nc+2):(Nc+Npred+2),2] .= CQuota[1]
	Catch[(Nc+2):(Nc+Npred+2),3] .= CQuota[2]
	Ft[Nc+2:end] .= Ftmp[end]

	# Preliminary calculations - Preparations of birth ogive
	P[1:Nc, 1:Amax] = Pmat[1:Nc, 1:Amax]
	P[Nc+1:Nc+Npred+1, 1:Amax] .= repeat(Pmat[Nc, 1:Amax], 1, Npred+1)'

	# Extract priors
	mub = priors[:,1]
	sdb = priors[:,2]
	b = [K,M,M0]

	#################################################
	# Initiate of N in year 0 (1945) - EQ 1 and 2
	N[1,:] = exp.(-[0:(Amax-1);] .* M) 		# Adults
	N[1,Amax] /= 1-em;						# Correct A+ group
	Nsum = sum(N[1,:])
	N[1,:] = K * N[1,:]/Nsum				# Normalize vector to K
	N1[1] = K								# Abundance of 1 year and older seals
	N0[1] = (1-em)/em0*K					# To balance natural mortality of 1+ group

	#################################################
	# Calculate population trajectory

	for i = 2:Nc+Npred+2
		N[i,1] = (N0[i-1]-Catch[i-1,2]) * em0	# 0-group from last year

		for j = 2:Amax
			N[i,j] = N[i-1,j-1] * (1-Catch[i-1,3]/N1[i-1]) * em	# Pro-rata distribution of catches
		end

		N[i,Amax] += N[i-1,Amax] * (1 - Catch[i-1,3]/N1[i-1]) * em	 # A+ group correction

		# Ensures that N > 0
		Nsum = sum(N[i,:])
		N1[i] = posfun(Nsum-Catch[i,3], 1.0) + Catch[i,3]

		N[i,:] = N1[i] * N[i,:] / Nsum

		# Recruitment equation
		N0[i] = 0.5 * Ft[i] * sum( P[i-1,:] .* N[i,:] )
		# Ensures that pup production is larger than pup catch
		N0[i] = posfun(N0[i]-Catch[i,2],1.0) + Catch[i,2]

		# Calculate the D(1+) statistic
		NTotPred = 0.0
		if Catch[i,1] == Catch[Nc+1,1]
			D1 = 1.0/(N1[i])
			N0CurrentYear = N0[i]
			N1CurrentYear = N1[i]
			NTotCurrentYear = (N0[i]+N1[i])
		end

		if Catch[i,1] == Catch[Nc+Npred+1,1]
			D1 *= N1[i]
			DNmax = N1[i]+N0[i]
			NTotPred = DNmax
		end
	end

	NTot = N0 + N1
	NTotmax = maximum(NTot)
	DNmax = NTotPred/NTotmax

	# Likelihood contribution from pup production estimates
	nll = -sum( logpdf.( Normal.(pupProductionData[1:Np,2], 
								 pupProductionData[1:Np,2] .* pupProductionData[1:Np,3]), 
				N0[floor.(Int, pupProductionData[1:Np,1]) .- (Cdata[1,1] - 2)] ) )
	# Likelihood contribution from prior distibutions - EQ 11
	nll -= sum( logpdf.( Normal.(priors[:,1], priors[:,2]), b ) )
	# Likelihood penalty to avoid negative N
	nll += 1e-3 * sum( ((abs.(N1[1:Nc] .- 1e4) .- (N1[1:Nc] .- 1e4))).^2 )

	return nll
end

# Test function to see if results are the same as rSPAMM
@time res = objective_function( [logK, Mtilde, M0tilde]; D )

@code_warntype objective_function( [logK, Mtilde, M0tilde]; D )

# Minimize using finite difference
result = optimize(x->objective_function(x; D), 
				[logK, Mtilde, M0tilde], BFGS());
result.minimizer, result.minimum

# Minize using autodiff
result_ad = optimize( x->objective_function(x; D), 
							[logK, Mtilde, M0tilde], BFGS(); autodiff = :forward)
result_ad.minimizer, result_ad.minimum

#endregion
