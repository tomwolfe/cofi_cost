require_relative 'spec_helper'

describe CofiCost do
	before :each do
		ratings = NArray[[5.0,4.0,0.0,0.0],[3.0,0.0,0.0,0.0],[4.0,0.0,0.0,0.0],[3.0,0.0,0.0,0.0],[3.0,0.0,0.0,0.0]]
		num_features = 3
		regularization = 1
		iterations = 10
		theta = NArray[[0.28544,-1.68427,0.26294],[0.50501,-0.45465,0.31746],[-0.43192,-0.47880,0.84671],[0.72860,-0.27189,0.32684]]
		features = NArray[[1.048686,-0.400232,1.194119],[0.780851,-0.385626,0.521198],[0.641509,-0.547854,-0.083796],[0.453618,-0.800218,0.680481],[0.937538,0.106090,0.361953]]
		@cofi = CofiCost.new(ratings, num_features, regularization, iterations, features, theta)
	end
  
	describe "#normalize_ratings" do
    it "subtracts the mean rating of a track from each of the tracks ratings" do
			# called in initilization
			@cofi.ratings_mean.should == NArray[[4.5],[3.0],[4.0],[3.0],[3.0]]
			@cofi.ratings_norm.should == NArray[[0.5,-0.5,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0]]
    end
  end

	describe "#partial_cost_calc(theta,features)" do
		it "calculates part of the cost" do
			@cofi.partial_cost_calc.to_a.should == [[0.78741733234,1.5906474134,0.0,0.0],[1.00942821458,0.0,0.0,0.0],[1.0838130652999998,0.0,-0.0,0.0],[1.65618956692,0.0,0.0,0.0],[0.18409856424,0.0,-0.0,0.0]]
		end
	end

	describe "#roll_up_theta_and_features" do
		it "returns a vector of @theta + @features" do
			@cofi.roll_up_theta_and_features.to_a.should == [0.28544,-1.68427,0.26294,0.50501,-0.45465,0.31746,-0.43192,-0.4788,0.84671,0.7286,-0.27189,0.32684,1.048686,-0.400232,1.194119,0.780851,-0.385626,0.521198,0.641509,-0.547854,-0.083796,0.453618,-0.800218,0.680481,0.937538,0.10609,0.361953]
		end
	end

	describe "#unroll_params_init_shape(x)" do
		before :each do
			@rolled = @cofi.roll_up_theta_and_features
		end
		it "unrolls @theta and @features back into it's original shape" do
			orig_theta, orig_features = @cofi.theta, @cofi.features
			@cofi.unroll_params_init_shape(@rolled)
			@cofi.theta.should == orig_theta
			@cofi.features.should == orig_features
		end
	end

	describe "#min_cost" do
		it "finds the lowest cost" do
			@cofi.min_cost
			@cofi.cost.should == 0.9885618408659723
		end
		it "calls #calc_predictions" do
			@cofi.should_receive(:calc_predictions)
			@cofi.min_cost
		end
	end

	describe "#calc_predictions" do
		it "calculates predictions" do
			@cofi.calc_predictions.to_a.should == [[5.78741733234,5.5906474134,5.24975512297,5.76317755204],[4.00942821458,3.73512194149,3.28867612346,3.84412424606],[5.083813065299999,4.54644840303,3.91428101676,4.58897159682],[4.65618956692,3.8089262381399998,3.76338775935,3.7704857568600003],[3.18409856424,3.54013784626,2.85073191967,3.77254609522]]
		end
	end

	describe "#unroll_params(v)" do
		it "unrolls v back to it's original @theta and @features dimensions" do
			@cofi.unroll_params(@cofi.roll_up_theta_and_features)[0].to_a.should == [[0.28544,-1.68427,0.26294],[0.50501,-0.45465,0.31746],[-0.43192,-0.4788,0.84671],[0.7286,-0.27189,0.32684]]
			@cofi.unroll_params(@cofi.roll_up_theta_and_features)[1].to_a.should == [[1.048686,-0.400232,1.194119],[0.780851,-0.385626,0.521198],[0.641509,-0.547854,-0.083796],[0.453618,-0.800218,0.680481],[0.937538,0.10609,0.361953]]
		end
	end
end
