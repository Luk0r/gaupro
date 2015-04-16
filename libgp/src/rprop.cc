// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include <stdlib.h>
#include <cmath>
#include <iostream>

#include <fstream> // logging

#include "rprop.h"
#include "gp_utils.h"


namespace libgp
{

void RProp::init ( double eps_stop, double Delta0, double Deltamin, double Deltamax, double etaminus, double etaplus )
{
    this->Delta0   = Delta0;
    this->Deltamin = Deltamin;
    this->Deltamax = Deltamax;
    this->etaminus = etaminus;
    this->etaplus  = etaplus;
    this->eps_stop = eps_stop;

}

std::vector<double> RProp::maximize ( GaussianProcess *gp, size_t n, bool verbose )
{
    int param_dim = gp->covf().get_param_dim();
    Eigen::VectorXd Delta = Eigen::VectorXd::Ones ( param_dim ) * Delta0;
    Eigen::VectorXd grad_old = Eigen::VectorXd::Zero ( param_dim );
    Eigen::VectorXd params = gp->covf().get_loghyper();
    Eigen::VectorXd params_old = gp->covf().get_loghyper();
    Eigen::VectorXd best_params = params;
    double best = log ( 0 );

    // logging
    std::ofstream fileGrad, fileParams, fileLik;
    fileGrad.open ( "/home/schmidan/data/Schaefertal_ASD_Bodenproben/ml_gp/py_scripts/logging/rprop_gradient.txt" );
    fileParams.open ( "/home/schmidan/data/Schaefertal_ASD_Bodenproben/ml_gp/py_scripts/logging/rprop_params.txt" );
    fileLik.open ( "/home/schmidan/data/Schaefertal_ASD_Bodenproben/ml_gp/py_scripts/logging/rprop_lik.txt" );




    std::vector<double> log_likelihood_curve;

    /*
    for (size_t i=0; i<n; ++i) {
      double lik = gp->log_likelihood();
      Eigen::VectorXd grad = -gp->log_likelihood_gradient();

      log_likelihood_curve.push_back(-lik);
      if (verbose) std::cout << i << " " << -lik <<  std::endl;

      grad_old = grad_old.cwiseProduct(grad);
      for (int j=0; j<grad_old.size(); ++j)
      {
    	if (grad_old(j) > 0) // no sign change
    	{
    		Delta(j) = std::min(Delta(j)*etaplus, Deltamax);
    	}

    	else if (grad_old(j) < 0) // sign change
    	{
    		Delta(j) = std::max(Delta(j)*etaminus, Deltamin);
    		grad(j) = 0;
    	}
    	params(j) += -Utils::sign(grad(j)) * Delta(j);
      }
      grad_old = grad;

      fileGrad << grad << std::endl;
      fileParams << params << std::endl;
      fileLik << lik << std::endl;

      if (grad_old.norm() < eps_stop) break;
      gp->covf().set_loghyper(params);
      if (lik > best) {
        best = lik;
        best_params = params;
      }
    }
    */






    for ( size_t i = 0; i < n; ++i )
    {
        double lik = gp->log_likelihood();
        if ( lik > best )
        {
            best = lik;
            best_params = params;
        }
        Eigen::VectorXd grad = -gp->log_likelihood_gradient();

        fileGrad << grad << std::endl;
        fileParams << params << std::endl;
        fileLik << lik << std::endl;

        log_likelihood_curve.push_back ( -lik );
        if ( verbose ) std::cout << i << " " << -lik <<  std::endl;

        grad_old = grad_old.cwiseProduct ( grad );
        for ( int j = 0; j < grad_old.size(); ++j )
        {
            if ( grad_old ( j ) > 0 ) // no sign change
            {
                Delta ( j ) = std::min ( Delta ( j ) * etaplus, Deltamax );
                params ( j ) += -Utils::sign ( grad ( j ) ) * Delta ( j );
            }

            else if ( grad_old ( j ) < 0 ) // sign change
            {
                Delta ( j ) = std::max ( Delta ( j ) * etaminus, Deltamin );
                params ( j ) += -Utils::sign ( grad ( j ) ) * Delta ( j );
                grad ( j ) = 0;
            }

            else
            {
                Delta ( j ) = std::max ( Delta ( j ) * etaminus, Deltamin );
                params ( j ) += -Utils::sign ( grad ( j ) ) * Delta ( j );
            }
        }
        grad_old = grad;
        params_old = params;

        if ( grad_old.norm() < eps_stop ) break;
        gp->covf().set_loghyper ( params );
    }




    gp->covf().set_loghyper ( best_params );

    fileGrad.close();
    fileParams.close();
    fileLik.close();

    return log_likelihood_curve;
}

}//namespace libgp

