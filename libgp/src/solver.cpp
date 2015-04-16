#include "solver.hpp"


void Optimizer::Bfgs()
{   
    p_solver.reset(new pwie::BfgsSolver);
    p_solver->settings.maxIter = maxIter;
    p_solver->solve(x, function_value, gradient_value); 
}

void Optimizer::Lbfgs()
{
    p_solver.reset(new pwie::LbfgsSolver);
    p_solver->settings.maxIter = maxIter;
    p_solver->solve(x, function_value, gradient_value); 
}


void Optimizer::Lbfgsb()
{   
    p_solver.reset(new pwie::LbfgsbSolver);
    p_solver->settings.maxIter = maxIter;
    p_solver->solve(x, function_value, gradient_value);
}


void Optimizer::Gd()
{
    p_solver.reset(new pwie::GradientDescentSolver);
    p_solver->settings.maxIter = maxIter;
    p_solver->solve(x, function_value, gradient_value);
}

void Optimizer::Cg()
{   
    p_solver.reset(new pwie::ConjugateGradientSolver);
    p_solver->settings.maxIter = maxIter;
    p_solver->solve(x, function_value, gradient_value);
}

void Optimizer::Newton()
{   
    p_solver.reset(new pwie::NewtonDescentSolver);
    p_solver->settings.maxIter = maxIter;
    p_solver->solve(x, function_value, gradient_value);
}

void Optimizer::Rprop()
{   
	p_solver.reset(new pwie::RpropSolver);
	p_solver->settings.maxIter = maxIter;
	p_solver->solve(x, function_value, gradient_value);
}

void Optimizer::Irpropplus()
{   
	p_solver.reset(new pwie::IrpropplusSolver);
	p_solver->settings.maxIter = maxIter;
	p_solver->solve(x, function_value, gradient_value);
}

void Optimizer::Irpropminus()
{   
	p_solver.reset(new pwie::IrpropminusSolver);
	p_solver->settings.maxIter = maxIter;
	p_solver->solve(x, function_value, gradient_value);
}
