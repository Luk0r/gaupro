#ifndef SOLVER_H
#define SOLVER_H

#include "gp.h"
#include <Eigen/Core>
#include <vector>
#include <functional>
#include <memory>

// solver
#include <Meta.h>
#include <BfgsSolver.h>
#include <LbfgsSolver.h>
#include <LbfgsbSolver.h>
#include <GradientDescentSolver.h>
#include <ConjugateGradientSolver.h>
#include <NewtonDescentSolver.h>
#include <Rprop.h>
#include <Irpropplus.h>
#include <Irpropminus.h>

class Solver
{
    
public:
    Solver(libgp::GaussianProcess& r_gp, size_t _maxIter = 100) : gp(r_gp), maxIter(_maxIter), p_solver(nullptr) {};
    
    void Bfgs();
    void Lbfgs();
    void Lbfgsb();
    void Cg();
    void Gd();
    void Newton();
	void Rprop();
	void Irpropplus();
	void Irpropminus();
    
    inline std::vector<double> get_FunctionValueHistory() {return p_solver->get_FunctionValueHistory(); };
    inline std::vector<pwie::Vector> get_x0History() {return p_solver->get_x0History(); };
    



private:
	libgp::GaussianProcess& gp;
	size_t maxIter;
    std::shared_ptr<pwie::ISolver> p_solver;


    
    decltype( gp.covf().get_loghyper() ) x = gp.covf().get_loghyper();
   

    pwie::function_t function_value = [&] ( const pwie::Vector &x ) -> double 
    {
        gp.covf().set_loghyper ( x );
        return -gp.log_likelihood();
    };
    

    pwie::gradient_t gradient_value = [&] ( const pwie::Vector x, pwie::Vector &grad ) -> void
    {
        gp.covf().set_loghyper ( x );
        grad = -gp.log_likelihood_gradient();
    };
    
};

#endif // SOLVER_H

