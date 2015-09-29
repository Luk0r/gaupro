#include "solver.hpp"


void Solver::Bfgs()
{   
    p_solver.reset(new pwie::BfgsSolver);
    p_solver->settings.maxIter = maxIter;
    p_solver->solve(x, function_value, gradient_value); 
}

void Solver::Lbfgs()
{
    p_solver.reset(new pwie::LbfgsSolver);
    p_solver->settings.maxIter = maxIter;
    p_solver->solve(x, function_value, gradient_value); 
}


void Solver::Lbfgsb()
{   
    p_solver.reset(new pwie::LbfgsbSolver);
    p_solver->settings.maxIter = maxIter;
    p_solver->solve(x, function_value, gradient_value);
}


void Solver::Gd()
{
    p_solver.reset(new pwie::GradientDescentSolver);
    p_solver->settings.maxIter = maxIter;
    p_solver->solve(x, function_value, gradient_value);
}

void Solver::Cg()
{   
    p_solver.reset(new pwie::ConjugateGradientSolver);
    p_solver->settings.maxIter = maxIter;
    p_solver->solve(x, function_value, gradient_value);
}

void Solver::Newton()
{   
    p_solver.reset(new pwie::NewtonDescentSolver);
    p_solver->settings.maxIter = maxIter;
    p_solver->solve(x, function_value, gradient_value);
}

void Solver::Rprop()
{   
	p_solver.reset(new pwie::RpropSolver);
	p_solver->settings.maxIter = maxIter;
	p_solver->solve(x, function_value, gradient_value);
}

void Solver::Irpropplus()
{   
	p_solver.reset(new pwie::IrpropplusSolver);
	p_solver->settings.maxIter = maxIter;
	p_solver->solve(x, function_value, gradient_value);
}

void Solver::Irpropminus()
{   
	p_solver.reset(new pwie::IrpropminusSolver);
	p_solver->settings.maxIter = maxIter;
	p_solver->solve(x, function_value, gradient_value);
}
