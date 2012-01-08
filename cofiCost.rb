require 'gsl'
require 'matrix'
require 'narray'

include GSL::MultiMin

class ConfiCost

	attr_accessor :ratings, :trackList, :numFeatures, :cost, :lambda # trackList: activerecord lookup of all track_ids
	attr_reader :booleanRated, :numTracks, :numUsers, :features, :theta, :ratingsMean, :ratingsNorm
	
	def initialize(ratings, trackList, numFeatures, lambda)
		@ratings = ratings
		@trackList = trackList
		@numFeatures = numFeatures
		@cost = 0
		@booleanRated = @ratings > 0
		@numTracks = @ratings.shape[1] # @ratings is users x tracks
		@numUsers = @ratings.shape[0]
		# set initial parameters
		@features = NArray.float(@numFeatures, @numTracks).randomn
		@theta = NArray.float(@numFeatures, @numUsers).randomn
		
		@ratingsMean = NArray.float(1, @numTracks).fill(0.0)
		@ratingsNorm = NArray.float(@numUsers, @numTracks).fill(0.0)
		@ratingsMean, @ratingsNorm = normalizeRatings
		@lambda = lambda
	end
	
	def normalizeRatings
		for i in 0..@numTracks-1 # sadly, @numTracks.each_index does not work with NArray yet
			trackRating = @ratings[true,i] # get all user ratings for track i (including unrated)
			booleanTrackRating = booleanRated[true,i] # get all user ratings that exist for track i
		    	@ratingsMean[i] = trackRating[booleanTrackRating].mean
		    	
		    	trackNorm = @ratingsNorm[true,i]
		    	trackNorm[booleanTrackRating] = trackRating[booleanTrackRating] - @ratingsMean[i]
		    	@ratingsNorm[true,i] = trackNorm
		end
		return @ratingsMean, @ratingsNorm
	end
	
	def unrollParams(v)
		v = v.to_na
		theta = v.slice(0..@theta.size-1).reshape(@theta.shape[0],true)
		features = v.slice(@theta.size..-1).reshape(@features.shape[0],true)
		return theta, features
	end
	
	def partialCostCalc(theta,features)
		(NArray.ref(NMatrix.ref(features) * NMatrix.ref(theta.transpose(1,0))) - @ratingsNorm) * @booleanRated
	end
	
	def minCost
		cost_f = Proc.new { |v|
			theta, features = unrollParams(v)
			# In octave:
			# 1/2 * sum(sum(((X * Theta.transpose - Y).*R).^2)) + lambda/2 * sum(sum((Theta).^2)) + lambda/2 * sum(sum((X).^2))
			(partialCostCalc(theta,features)**2).sum + @lambda/2 * (features**2).sum
		}
		cost_df = Proc.new { |v, df|
			theta, features = unrollParams(v)
			# In octave:
			# xgrad = ((X * Theta.transpose - Y).* R) * Theta + lambda * X # X_grad
			# thetagrad = ((X * Theta.transpose - Y).* R).transpose * X + lambda * Theta
			
			# I realize this is a hack. I'm not totally sure why or how but just setting
			# df = NArray.hcat(dfzero,dfone) results in no steps being made in gradient descent.
			# ideas/suggestions welcome :)
			dfzero = (NArray.ref(NMatrix.ref(partialCostCalc(theta,features)) * NMatrix.ref(theta)) + @lambda * features).flatten
			dfone = (NArray.ref(NMatrix.ref((partialCostCalc(theta,features)).transpose(1,0)) * NMatrix.ref(features)) + @lambda * theta).flatten
			dfcomp = NArray.hcat(dfzero,dfone)
			for i in 0..dfcomp.size-1	# again .each_index does not yet work with NArray
				df[i] = dfcomp[i]
			end
		}
		
		# roll up theta and features together
		# (oddly, NArray objects created don't seem to recognize the hcat method
		# 	added to the open class NArray
		#	x = GSL:: Vector.alloc(@theta.reshape(true,1).hcat(@features.reshape(true,1)))
		#		will fail)
		#		I don't understand why this is/how to fix it.
		thetaReshaped = @theta.reshape(true)
		featuresReshaped = @features.reshape(true)
		rolled = NArray.hcat(thetaReshaped,featuresReshaped)
		x = GSL:: Vector.alloc(rolled) # starting point
		cost_func = Function_fdf.alloc(cost_f, cost_df, x.size)

		# TODO: figure out which algorithm to use
		# http://www.gnu.org/software/gsl/manual/html_node/Multimin-Algorithms-with-Derivatives.html
		minimizer = FdfMinimizer.alloc("conjugate_fr", x.size)
		minimizer.set(cost_func, x, 0.01, 1e-4)
		
		iter = 0
		begin
			iter += 1
			status = minimizer.iterate()
			status = minimizer.test_gradient(1e-3)
			if status == GSL::SUCCESS
				puts("Minimum found at")
			end
			x = minimizer.x
			f = minimizer.f
			printf("%5d %.5f %.5f %10.5f\n", iter, x[0], x[1], f)
		end while status == GSL::CONTINUE and iter < 10
		# return x
	end

end

class NArray
	class << self
		def cat(dim=0, *narrays)
		      raise ArgumentError, "'dim' must be an integer" unless dim.is_a?(Integer)
		      raise ArgumentError, "must have narrays to cat" if narrays.size == 0
		      new_typecode = narrays.map(&:typecode).max
		      narrays.uniq.each {|narray| narray.newdim!(dim) if narray.shape[dim].nil? }
		      shapes = narrays.map(&:shape)
		      new_dim_size = shapes.inject(0) {|sum,v| sum + v[dim] }
		      new_shape = shapes.first.dup
		      new_shape[dim] = new_dim_size
		      narr = NArray.new(new_typecode, *new_shape)
		      range_cnt = 0
		      narrays.zip(shapes) do |narray, shape|
			index = shape.map {true}
			index[dim] = (range_cnt...(range_cnt += shape[dim]))
			narr[*index] = narray
		      end
		      narr
		end
		def vcat(*narrays) ; cat(1, *narrays) end
		def hcat(*narrays) ; cat(0, *narrays) end
	end
end

ratings = NArray.int(6,5).indgen(0,2)
trackList = Array.new(ratings.shape[1])
numFeatures = 2
lambda = 1
g = ConfiCost.new(ratings, trackList, numFeatures, lambda)
g.minCost
