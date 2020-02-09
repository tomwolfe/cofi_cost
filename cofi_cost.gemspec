Gem::Specification.new do |s|
  s.name        = 'cofi_cost'
  s.version     = '0.0.10'
  s.date        = '2020-02-09'
  s.summary     = "Collaborative filtering"
  s.description = "Playground for collaborative filtering in Ruby using NArray and rb-gsl."
  s.authors     = ["Thomas Wolfe"]
  s.email       = 'tomwolfe@gmail.com'
  s.files       = `git ls-files`.split("\n")
	s.homepage    = 'http://github.com/tomwolfe/cofi_cost'
  s.add_runtime_dependency 'gsl'
  s.add_runtime_dependency 'narray'
  s.license = 'MIT'
  s.required_ruby_version = '>= 1.9.2'
  s.requirements << 'libgsl0-dev'
end
