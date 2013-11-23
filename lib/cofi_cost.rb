require 'gsl'
require 'matrix'
require 'narray'

include GSL::MultiMin

class CofiCost

	attr_accessor :ratings, :num_features, :regularization, :iterations, :features, :theta, :max_rating, :min_rating
	attr_reader :boolean_rated, :num_tracks, :num_users, :ratings_mean, :ratings_norm, :predictions, :cost
	
	def initialize(ratings, num_features = 2, regularization = 1, iterations = 10, max_rating = 5, min_rating = 0, features = nil, theta = nil)
		@ratings = ratings.to_f	# make sure it's a float for correct normalization
		@num_features = num_features
		@cost = 0
		@max_rating = max_rating
		@min_rating = min_rating
		@boolean_rated = @ratings > 0 # return 0 for all rated and 1 for all unrated
		@boolean_unrated = @boolean_rated.eq 0 # return 1 for all unrated and 0 for all unrated
		@num_tracks = @ratings.shape[1] # @ratings is users x tracks
		@num_users = @ratings.shape[0]
		# set initial parameters
		# allow theta/features to be set for testing
		if features.nil? then @features = NArray.float(@num_features, @num_tracks).randomn else @features = features end
		if theta.nil? then @theta = NArray.float(@num_features, @num_users).randomn else @theta = theta end
		@ratings_mean = NArray.float(1, @num_tracks).fill(0.0)
		@ratings_norm = NArray.float(@num_users, @num_tracks).fill(0.0)
		@ratings_mean, @ratings_norm = normalize_ratings
		@regularization = regularization
		@predictions = nil
		@iterations = iterations
	end
	
	def normalize_ratings
		for i in 0..@num_tracks-1 # sadly, @num_tracks.each_index does not work with NArray yet
			track_rating = @ratings[true,i] # get all user ratings for track i (including unrated)
			boolean_track_rating = @boolean_rated[true,i] # get all user ratings that exist for track i
		    	track_rating_boolean = track_rating[boolean_track_rating]
		    	if track_rating_boolean.size == 0
		    	  @ratings_mean[i] = 0
		    	else
		    	  @ratings_mean[i] = track_rating_boolean.mean
		    	end
		    	
		    	track_norm = @ratings_norm[true,i]
		    	track_norm[boolean_track_rating] = track_rating[boolean_track_rating] - @ratings_mean[i]
		    	@ratings_norm[true,i] = track_norm
		end
		return @ratings_mean, @ratings_norm
	end
	
	def unroll_params(v)
		v = v.to_na
		theta = v.slice(0..@theta.size-1).reshape(@theta.shape[0],true)
		features = v.slice(@theta.size..-1).reshape(@features.shape[0],true)
		return theta, features
	end
	
	def partial_cost_calc(theta=@theta,features=@features)
		(NArray.ref(NMatrix.ref(features) * NMatrix.ref(theta.transpose(1,0))) - @ratings_norm) * @boolean_rated
	end
	
	def roll_up_theta_and_features
		theta_reshaped = @theta.reshape(true)
		features_reshaped = @features.reshape(true)
		rolled = NArray.hcat(theta_reshaped,features_reshaped)
		GSL::Vector.alloc(rolled) # starting point
	end
	
	def unroll_params_init_shape(x)
		theta, features = unroll_params(x)
		@theta = theta.reshape(@theta.shape[0],true)
		@features = features.reshape(@features.shape[0],true)
	end
	
	def min_cost
		cost_f = Proc.new { |v|
			theta_l, features_l = unroll_params(v)
			# In octave:
			# 1/2 * sum(sum(((X * Theta.transpose - Y).*R).^2)) + regularization/2 * sum(sum((Theta).^2)) + regularization/2 * sum(sum((X).^2))
			0.5 * (partial_cost_calc(theta_l,features_l)**2).sum + @regularization/2 * (features_l**2).sum + @regularization/2 * (theta_l**2).sum
		}
		cost_df = Proc.new { |v, df|
			theta_l, features_l = unroll_params(v)
			# In octave:
			# xgrad = ((X * Theta.transpose - Y).* R) * Theta + regularization * X # X_grad
			# thetagrad = ((X * Theta.transpose - Y).* R).transpose * X + regularization * Theta
			
			# I realize this is a hack. I'm not totally sure why or how but just setting
			# df = NArray.hcat(dfzero,dfone).to_gv results in no steps being made in gradient descent.
			# ideas/suggestions welcome :)
			dfzero = (NArray.ref(NMatrix.ref(partial_cost_calc(theta_l,features_l)) * NMatrix.ref(theta_l)) + @regularization * features_l).flatten
			dfone = (NArray.ref(NMatrix.ref((partial_cost_calc(theta_l,features_l)).transpose(1,0)) * NMatrix.ref(features_l)) + @regularization * theta_l).flatten
			dfcomp = NArray.hcat(dfzero,dfone)
			for i in 0..dfcomp.size-1	# again .each_index does not yet work with NArray
				df[i] = dfcomp[i]
			end
		}
		
		x = roll_up_theta_and_features
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
		end while status == GSL::CONTINUE and iter < @iterations
		
		unroll_params_init_shape(x)
		@cost = f
		@predictions = calc_predictions
	end
	
	def calc_predictions
		predicts = NArray.ref(NMatrix.ref(@features) * NMatrix.ref(@theta.transpose(1,0))) + @ratings_mean
		set_max_min_predictions(predicts)
	end

	def set_max_min_predictions(predicts)
		over_max = predicts > @max_rating
		under_min = predicts < @min_rating
		predicts[over_max] = @max_rating
		predicts[under_min] = @min_rating
		predicts
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
