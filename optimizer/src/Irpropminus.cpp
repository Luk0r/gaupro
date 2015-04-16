#include "Irpropminus.h"
#include <limits>

#include <iostream>

namespace pwie
{
	
	IrpropminusSolver::IrpropminusSolver() : ISolver()
	{
		
	}
	
	void IrpropminusSolver::internalSolve(Vector & x0,
										 const function_t & FunctionValue,
									  const gradient_t & FunctionGradient,
									  const hessian_t & FunctionHessian)
	{
		//helper function
		auto sign = [](double x) -> double
		{
			if (x>0) return 1.0;
			if (x<0) return -1.0;
			return 0.0;
		};
		
		const size_t DIM = x0.rows();
		Vector Delta = Eigen::VectorXd::Ones ( DIM ) * Delta0; //step size
		Vector grad = Eigen::VectorXd::Zero ( DIM );
		Vector grad_old = Eigen::VectorXd::Zero ( DIM );
		
		Vector params = x0; // hyperparameter
		Vector best_params = params;
		
		double best = std::numeric_limits<double>::infinity();
		
		size_t iter = 0;
	
		do
		{
			FunctionGradient(params, grad); // calc gradient
			auto FunctionValue_this =  FunctionValue(params);
			
			grad_old = grad_old.cwiseProduct ( grad );
			auto params_new = params;
			for ( int j = 0; j < grad_old.size(); ++j )
			{
				if ( grad_old ( j ) > 0 ) // no sign change
				{
					Delta ( j ) = std::min ( Delta ( j ) * etaplus, Deltamax );
				}
				
				else if ( grad_old ( j ) < 0 ) // sign change
				{
					Delta ( j ) = std::max ( Delta ( j ) * etaminus, Deltamin );
					grad ( j ) = 0;
				}

				params_new ( j ) +=  -sign(grad (j))  * Delta (j);
			}
			
			params = params_new;
			
			//std::cout << best << std::endl;
			//std::cout <<  iter << " "<< FunctionValue_this << std::endl;
			FunctionValueHistory.push_back(FunctionValue_this);
			x0History.push_back(x0);
			
			if ( FunctionValue_this < best )
			{
				best = FunctionValue_this;
				best_params = params;
			}
			
			grad_old = grad;
			
			++iter;
		}
		while((grad_old.lpNorm<Eigen::Infinity>() > settings.gradTol) && (iter < settings.maxIter)); // if ( grad_old.norm() < eps_stop ) break;
		//while((iter < settings.maxIter)); // if ( grad_old.norm() < eps_stop ) break;
		
		// save best hyperparameters
		double temp = FunctionValue(best_params);
	}
	
}//namespace pwie
