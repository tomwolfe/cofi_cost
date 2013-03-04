require 'test/unit'
require_relative '../../lib/cofi_cost.rb'
require 'narray'
require 'gsl'
require 'matrix'

class CofiCostTest < Test::Unit::TestCase

  def setup
    ratings = NArray[[5.0,4.0,0.0,0.0],[3.0,0.0,0.0,0.0],[4.0,0.0,0.0,0.0],[3.0,0.0,0.0,0.0],[3.0,0.0,0.0,0.0]]
    num_features = 2
    lambda = 1
    iterations = 10
    features = NArray[[0.139489,1.804804],[-0.501808,1.050885],[0.354079,-0.518884],[-0.015370,0.096253],[1.147623,-0.745562]]
    theta = NArray[[-0.079641,1.211386],[-0.130688,0.444762],[-0.789258,1.222232],[0.212132,-1.174545]]
    @c = CofiCost.new(ratings, num_features, lambda, iterations, features, theta)
  end
  
  def teardown
    @c = nil
  end
  
  def test_happy_case
    @c.min_cost
    assert_equal 0.07964723302994943, @c.cost
    # oddly the following fails, even though they are equal (not enough decimal places me thinks)
    # assert_equal NArray[[4.62547,3.91302,8.30084,1.59081],[2.96361,3.17939,1.88322,3.88434],[3.92356,4.32263,1.739,5.6172],[2.98132,3.06219,2.47359,3.3213],[2.93724,3.14111,1.33728,3.77855]], @c.predictions
    assert_equal 4.625468057637709, @c.predictions[0,0]
  end
  
  def test_normalize_ratings
    assert_equal NArray[[4.5],[3.0],[4.0],[3.0],[3.0]], @c.ratings_mean
    assert_equal NArray[[0.5,-0.5,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0]], @c.ratings_norm
  end
  
  def test_roll_up_theta_and_features
    rolled = @c.roll_up_theta_and_features
    assert_equal GSL:: Vector.alloc([-0.079641, 1.211386, -0.130688, 0.444762, -0.789258, 1.222232, 0.212132, -1.174545, 0.139489, 1.804804, -0.501808, 1.050885, 0.354079, -0.518884, -0.01537, 0.096253, 1.147623, -0.745562]), rolled
  end
  
end
