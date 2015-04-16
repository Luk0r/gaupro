/*
 * cg.h
 *
 *  Created on: Feb 22, 2013
 *      Author: Joao Cunha <joao.cunha@ua.pt>
 */

#ifndef CG_H_
#define CG_H_

#include "gp.h"
#include <vector>

namespace libgp
{

class CG
{
public:
	CG();
	virtual ~CG();

	/**
	* maximizes the marginal likelihood
	* @return the marginal likelihood curve over the iterations
	*/
	std::vector<double> maximize(GaussianProcess* gp, size_t n=100, bool verbose=1);
};

}

#endif /* CG_H_ */
