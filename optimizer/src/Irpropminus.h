#ifndef IRPROPMINUS_H
#define IRPROPMINUS_H

#include "ISolver.h"

/*
 * typedef std::function<double(const Eigen::VectorXd & x)> function_t;
 * typedef std::function<void(const Eigen::VectorXd & x, Eigen::VectorXd & gradient)> gradient_t;
 * typedef std::function<void(const Eigen::VectorXd & x, Eigen::MatrixXd & hessian)> hessian_t;
 */
namespace pwie
{
	
	class IrpropminusSolver : public ISolver
	{
	public:
		IrpropminusSolver();
		void internalSolve(Vector & x0,
						   const function_t & FunctionValue,
					 const gradient_t & FunctionGradient,
					 const hessian_t & FunctionHessian = EMPTY_HESSIAN);
		
		//default values
		double eps_stop = 0.0;
		double Delta0 = 0.1; //starting speed
		double Deltamin = 1e-6; //min speed
		double Deltamax = 50; //max speed
		double etaminus = 0.5; //breaking
		double etaplus = 1.2; //acceleration
	};
	
	
} // namespace pwie


#endif // IRPROPMINUS_H
