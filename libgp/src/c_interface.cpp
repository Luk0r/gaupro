#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <cstdint>

#include <gp.h>
#include <rprop.h>
#include <cg.h>
#include <solver.hpp>

#include <utility> // pair
#include <cmath> // for NAN

using std::cout;
using std::endl;
using std::string;
using std::vector;
using std::cerr;

using namespace libgp;

extern "C"
{
    
void test()
{
    cout << "Hello World! 2" << endl;
}

double add(double a, double b)
{
    return a+b;
}

void init(int dim, std::string covf)
{
    cout << "dim: " << dim << "covf: " << covf << endl;
    
}

void get_array(double* arr, int len)
{
    for(int i=0; i<len;++i)
    {
        cout << arr[i] << endl;
    }
    arr[0] = 99.9;
}

void x_train(double* x, int dims, int* shape)
{
    cout << "dims = " << dims << endl;
    //cout << "dims: " << endl;
    for(int i =0; i< dims; ++i)
    {
        cout << "i " << i << " " << shape[i] << endl;
    }
    
    cout << "1. " << x[0] << endl;
    cout << "2. " << x[1] << endl;
    cout << "3. " << x[2] << endl;
    // print 1., 2. and last element per line
    
    int counter =0;
    for(int j=0; j<shape[0]*shape[1]; j+=368)
    {
        cout << counter << "   " << x[j] << " " << x[j+1] << " " << x[j+shape[1]-1] << endl;
        ++counter;
    }
    
    
    
}


GaussianProcess*  gp_new(unsigned int input_dim, char* covf)
{
    //debug:
    std::cout << covf << std::endl;
    
    GaussianProcess* gp = new GaussianProcess(input_dim, string(covf)) ;
    if(gp != nullptr)
    {
        cout << "gp = " << gp << endl;
        cout << "gp = " << uint64_t(gp) << endl;
        cout << "check" << endl;
        //cout << "*gp = " << *gp << endl;
        return gp;
    }
    else
    {
        cout << "ERROR in gp_new(unsigned int ndim, char* covf)" << endl;
        return nullptr;
    }
}

/*
    * uint64_t  gp_new(unsigned int input_dim, char* covf)
    * //GaussianProcess* gp_new(unsigned int input_dim, char* covf)
    * {
    *    //debug:
    *    std::cout << covf << std::endl;
    * 
    *    GaussianProcess* gp = new GaussianProcess(input_dim, string(covf)) ;
    *    if(gp != nullptr)
    *    {
    *        cout << "gp = " << gp << endl;
    *        cout << "gp = " << uint64_t(gp) << endl;
    *        //cout << "*gp = " << *gp << endl;
    *        //return uint64_t(gp);
    *        cout << "fixed return value = " << uint64_t(4294967294) << endl;
    *        return(uint64_t(4294967294));
    *        //return gp;
}
else
{
cout << "ERROR in gp_new(unsigned int ndim, char* covf)" << endl;
//return nullptr;
return 0;
}
}
*/





//void gp_add_train(GaussianProcess* gp_ptr, const double x[], double y)
void gp_add_train(GaussianProcess* gp_ptr, const double* x, int ndim, int* shape, double* y)
{
    if(ndim != 2)
    {
        cerr << "ERROR: dims of training data not 2" << endl;
        exit(0);
    }
    else
    {
        if(gp_ptr->get_input_dim() == shape[1])
        {
            int counter =0;
            for(int j=0; j<shape[0]*shape[1]; j+=shape[1])
            {
                //debug:
                //cout << counter << "   " << x[j] << " " << x[j+1] << " " << x[j+shape[1]-1] << endl;
                
                gp_ptr->add_pattern(&x[j], y[counter]);
                
                ++counter;
            }
        }
        else
        {
            cerr << "ERROR: gp_ptr->input_dim != shape[1]" << endl;
        }
    }
}



class Solvers
{

public:
	Solvers(GaussianProcess* gp_ptr, int iters) : optimizer(*gp_ptr, iters)
	{		
		SolverStrings.push_back("rprop_old");
		SolverStrings.push_back("irprop+_old");
		SolverStrings.push_back("cg_old");

		
		SolverPairs.push_back(std::make_pair("irpropplus", [this](){optimizer.Irpropplus();} )); // iRprop+ needs to be 1. for default selection
		SolverPairs.push_back(std::make_pair("irpropminus", [this](){optimizer.Rprop();} ));
		SolverPairs.push_back(std::make_pair("rprop", [this](){optimizer.Rprop();} ));
		SolverPairs.push_back(std::make_pair("bfgs", [this](){optimizer.Bfgs();} ));
		SolverPairs.push_back(std::make_pair("lbfgs", [this](){optimizer.Lbfgs();} ));
		SolverPairs.push_back(std::make_pair("lbfgsb", [this](){optimizer.Lbfgsb();} ));
		SolverPairs.push_back(std::make_pair("cg", [this](){optimizer.Cg();} ));
		SolverPairs.push_back(std::make_pair("gd", [this](){optimizer.Gd();} ));
	}
	
	void showSolvers()
	{
		std::cout << "available solvers are: " << std::endl;
		std::cout << "-----------------------" << std::endl;
		for(auto& s:SolverStrings)
		{
			std::cout << s << std::endl;
		}
		for(auto& s:SolverPairs)
		{
			std::cout << s.first << std::endl;
		}
		std::cout << "-----------------------" << std::endl;
	}
	
	Optimizer selectSolver(std::string s)
	{
		bool match = false;
		
		for(auto& t : SolverPairs)
		{
			if(t.first.compare(s) == 0)
			{
				std::cout << "selected optimizer: "  << t.first << std::endl;
				t.second();
				match = true;
			}
		}
		
		if(match == false)
		{
			std::cout << "no matching solver found, using default optimizer: iRprop+ (irpropplus)" << std::endl;
			SolverPairs[0].second();
		}
		
		
		return optimizer;
	}
	
	inline std::string at(size_t i) { return SolverStrings.at(i);}
	
		
	
private:
	Optimizer optimizer;
	std::vector<std::string> SolverStrings;
	std::vector<std::pair< std::string, std::function<void()> >> SolverPairs;
};

double* gp_optimize(GaussianProcess* gp_ptr, char* optimizer_c, int iters, double eps_stop = 0.0)
{
	Solvers s(gp_ptr, iters);
	s.showSolvers();
	
	std::string optimizerString(optimizer_c);
	std::transform(optimizerString.begin(), optimizerString.end(), optimizerString.begin(), ::tolower);
	
	auto optimizer = s.selectSolver(optimizerString);
	auto likelihood_curve = optimizer.get_FunctionValueHistory();
	
	double* likelihood_arr = new double[iters];


	for(size_t i=0; i<iters; ++i)
	{
		likelihood_arr[i] = likelihood_curve[i];
	}

	if(likelihood_curve.size() < iters)
	{
		for(int i=likelihood_curve.size(); i<iters; ++i)
		{
			likelihood_arr[i] = likelihood_curve.back();
		}
	}

	return likelihood_arr;
 
    
    /*
    if( optimizer.compare(s.at(0)) == 0)
    {
        cout << "opti is: "<< s.at(0) << endl;
        
        RProp* rprop_ptr = new RProp();
        rprop_ptr->init(eps_stop);
        
        vector<double> likelihood_curve = rprop_ptr->maximize(gp_ptr, iters, 0);
        double* likelihood_arr = new double[iters];
        
        for(size_t i=0; i<likelihood_curve.size(); ++i)
        {
            likelihood_arr[i] = likelihood_curve[i];
            
            //debug:
            //cout << likelihood_curve[i] << endl;
        }
        
        if(likelihood_curve.size() < iters)
        {
            
            for(int i=likelihood_curve.size(); i<iters; ++i)
            {
                likelihood_arr[i] = likelihood_curve.back();
            }
        }
        //debug
        //cout << "likelihood_curve.size() = " << likelihood_curve.size() << endl;
        
        delete rprop_ptr;
        return likelihood_arr;
    }
    
    else if( optimizer.compare(s.at(1)) == 0)
    {
		cout << "opti is: "<< s.at(1) << endl;
        
        
        
        iRPropPlus* iRPropPlus_ptr = new iRPropPlus();
        iRPropPlus_ptr->init(eps_stop);
        
        vector<double> likelihood_curve = iRPropPlus_ptr->maximize(gp_ptr, iters, 0);
        double* likelihood_arr = new double[iters];
        
        for(size_t i=0; i<likelihood_curve.size(); ++i)
        {
            likelihood_arr[i] = likelihood_curve[i];
            
            //debug:
            //cout << likelihood_arr[i] << endl;
        }
        
        if(likelihood_curve.size() < iters)
        {
            
            for(int i=likelihood_curve.size(); i<iters; ++i)
            {
                likelihood_arr[i] = likelihood_curve.back();
            }
        }
        
        
        cout << "likelihood_curve.size() = " << likelihood_curve.size() << endl;
        delete iRPropPlus_ptr;
        return likelihood_arr;
    }
    
    else if( optimizer.compare(s.at(2)) == 0)
    {
		cout << "opti is: " << s.at(2) << endl;
        
        CG* cg_ptr = new CG();
        
        vector<double> likelihood_curve = cg_ptr->maximize(gp_ptr, iters, 0);
        double* likelihood_arr = new double[iters];
        
        for(size_t i=0; i<likelihood_curve.size(); ++i)
        {
            likelihood_arr[i] = likelihood_curve[i];
            
            //debug:
            //cout << likelihood_arr[i] << endl;
        }
        
        if(likelihood_curve.size() < iters)
        {
            
            for(int i=likelihood_curve.size(); i<iters; ++i)
            {
                likelihood_arr[i] = likelihood_curve.back();
            }
        }
        
        
        cout << "likelihood_curve.size() = " << likelihood_curve.size() << endl;
        delete cg_ptr;
        return likelihood_arr;
    }   
    
    
    else
    {
        cout << "ERROR: neither RPROP nor CG not iRPROP+ was selected" << endl;
        double* likelihood_arr = new double[iters];
        for(size_t i=0; i<iters; ++i)
        {
            likelihood_arr[i] = 0;
            
            //debug:
            //cout << likelihood_arr[i] << endl;
        }
        return likelihood_arr;
    }
    */
    
}

double* gp_predict_value(GaussianProcess* gp_ptr, const double* x, int ndim, int* shape)
{
    vector<double> values;
    vector<double> variance;
    
    if(ndim == 1)
    {
        values.push_back( gp_ptr->f(&x[0]) );
        variance.push_back( gp_ptr->var(&x[0]));
    }
    else if(ndim == 2 )
    {
        int counter =0;
        
        for(int j=0; j<shape[0]*shape[1]; j+=shape[1])
        {
            //debug:
            //cout << counter << " " << j << "   " << x[j] << " " << x[j+1] << " " << x[j+2] << " "  << x[j+shape[1]-1] << endl;
            
            values.push_back( gp_ptr->f(&x[j]) );
            variance.push_back( gp_ptr->var(&x[j]));
            
            ++counter;
        }
    }
    
    else if(ndim > 2)
    {
        cerr << "ERROR: dims of prediction greater than 2" << endl;
        exit(0);
        
    }
    
    /*
        *    if(shape[1] != gp_ptr->get_input_dim())
        *    {
        *        cerr << "ERROR: prediction dimensions do not match trainign dims" << endl;
        *        exit(0);
}
*/
    
    //cout << "values.size() + variance.size() " << values.size() + variance.size() << endl;
    double* values_variance = new double[ values.size() + variance.size() ];
    //cout << " values.size() + variance.size() = " << values.size() + variance.size() << endl;
    for(int i=0; i < values.size() ;++i) // care: only half of the array lengths as index range!!!
    {
        values_variance[i] = values[i];
        values_variance[values.size() +i] = variance[i];
        //cout << i << " " << values_variance[i]  <<  " " << values_variance[values.size()+i] << endl;
    }
    
    
    return values_variance;
}

int gp_get_loghyper_len(GaussianProcess* gp_ptr)
{
    cout << "gp_get_loghyper_len(GaussianProcess* gp_ptr) 1" << endl;
    Eigen::VectorXd loghyperparam = gp_ptr->covf().get_loghyper(); // core dump here
    cout << "gp_get_loghyper_len(GaussianProcess* gp_ptr) 2" << endl;
    cout << loghyperparam.size() << endl;
    return loghyperparam.size();
    
    //original:
    //return gp_ptr->covf().get_loghyper().size();
}


double* gp_get_loghyper(GaussianProcess* gp_ptr, int l)
{
    Eigen::VectorXd loghyperparam = gp_ptr->covf().get_loghyper();
    
    //cout << "zzzzzzzzzzzz" << endl;
    //cout << loghyperparam << endl;
    
    int len = loghyperparam.size();
    if(len == l)
    {
        double* logHyperParam_array = new double[len];
        
        for(int i=0; i<loghyperparam.size(); ++i)
        {
            logHyperParam_array[i] = loghyperparam[i];
        }
        
        //debug:
        /*
            *        cout << "len = " << len << endl;
            *        cout << "sdfsjklfhjsdkfsdhkfksdjhfjks" << endl;
            *        for(int i=0; i<len; ++i)
            *        {
            *            cout << logHyperParam_array[i] << endl;
    }
    */
        
        
        return logHyperParam_array;
    }
    
    
    else
    {
        cout << "ERROR, lengths of hyperparameter vector does not match with given value" << endl;
        return nullptr;
    }
}

int gp_get_loghyperparam_dim(GaussianProcess* gp_ptr)
{
    return int(gp_ptr->covf().get_param_dim());
}

void gp_set_loghyper(GaussianProcess* gp_ptr, double* loghyperparam, int len)
{
    //cout << "gp_ptr->covf().get_param_dim() = " << gp_ptr->covf().get_param_dim() << endl;
    //cout << "len = " << len  << endl;
    if(len == gp_ptr->covf().get_param_dim())
    {
        gp_ptr->covf().set_loghyper(loghyperparam);
    }
    else
    {
        cout << "ERROR: dims of hyperparams do not match" << endl;
    }
}

double gp_get_loglikelihood(GaussianProcess* gp_ptr)
{
    return gp_ptr->log_likelihood();
}

    
    
}// extern "C"




