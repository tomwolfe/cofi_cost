     require 'gsl'
     require 'matrix'
     include GSL::MultiMin

     my_f = Proc.new { |v|
       x = v[0]; y = v[1]
       10.0*(x - 1)*(x - 1) + 20.0*(y - 2)*(y - 2) + 30.0
     }

     my_df = Proc.new { |v, df|
       x = v[0]; y = v[1]
       df[0] = 20.0*(x-1)
       df[1] = 40.0*(y-2)
     }

     my_func = Function_fdf.alloc(my_f, my_df, 2)
     #my_func.set_params([1.0, 2.0])      # parameters

     # x = Vector.alloc(5.0, 7.0)          # starting point
	x = GSL:: Vector.alloc(5.0, 7.0)
     minimizer = FdfMinimizer.alloc("conjugate_fr", 2)
     minimizer.set(my_func, x, 0.01, 1e-4)

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
     end while status == GSL::CONTINUE and iter < 100
