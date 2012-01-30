require 'test/unit'
#require File.dirname(__FILE__) + '/cofiCost.rb'
require_relative '../cofiCost.rb'
require 'narray'
require 'gsl'
require 'matrix'

class CofiCostTest < Test::Unit::TestCase

  def setup
    ratings = NArray[[5.0,4.0,0.0,0.0],[3.0,0.0,0.0,0.0],[4.0,0.0,0.0,0.0],[3.0,0.0,0.0,0.0],[3.0,0.0,0.0,0.0]]
    num_features = 2
    lambda = 1
    features = NArray[[0.139489,1.804804],[-0.501808,1.050885],[0.354079,-0.518884],[-0.015370,0.096253],[1.147623,-0.745562]]
    theta = NArray[[-0.079641,1.211386],[-0.130688,0.444762],[-0.789258,1.222232],[0.212132,-1.174545]]
    @c = CofiCost.new(ratings, num_features, lambda, features, theta)
  end
  
  def teardown
    @c = nil
  end  
  
  def test_normalize_ratings
    assert_equal NArray[[4.5],[3.0],[4.0],[3.0],[3.0]], @c.ratings_mean
    assert_equal NArray[[0.5,-0.5,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0]], @c.ratings_norm
  end
  
  def test_roll_up_theta_and_features
    rolled = @c.roll_up_theta_and_features
    assert_equal GSL:: Vector.alloc([-0.079641, 1.211386, -0.130688, 0.444762, -0.789258, 1.222232, 0.212132, -1.174545, 0.139489, 1.804804, -0.501808, 1.050885, 0.354079, -0.518884, -0.01537, 0.096253, 1.147623, -0.745562]), rolled
  end
  
 # def test_unroll_params
  #  theta, features = @c.unroll_params
   # assert_equal 4, theta
    #assert_equal 4, features
  #end
  
end
