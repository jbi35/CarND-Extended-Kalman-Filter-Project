#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  // make some sanity checks
  assert(estimations.size() > 0);
  assert(estimations.size() == ground_truth.size());

  VectorXd rmse(4);
  rmse << 0.0, 0.0, 0.0, 0.0;

	//accumulate squared residuals
	for(unsigned int i=0; i < estimations.size(); ++i){

		VectorXd residual = estimations[i] - ground_truth[i];

		//coefficient-wise multiplication
		residual = residual.array()*residual.array();
		rmse += residual;
	}

	//calculate the mean
	rmse = rmse/estimations.size();

	//calculate the squared root
	rmse = rmse.array().sqrt();

	//return the result
	return rmse;

}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {

  MatrixXd Hj(3,4);
	//recover state parameters
	const double px = x_state(0);
	const double py = x_state(1);
	const double vx = x_state(2);
	const double vy = x_state(3);

	//pre-compute terms
	double c1 = px*px+py*py;
  // safety check
  if(fabs(c1) < 1.0e-08){
    c1 = 1.0e-08;
  }
	const double c2 = sqrt(c1);
	const double c3 = (c1*c2);

	//compute the Jacobian
	Hj << (px/c2), (py/c2), 0, 0,
		   -(py/c1), (px/c1), 0, 0,
		    py*(vx*py - vy*px)/c3, px*(px*vy - py*vx)/c3, px/c2, py/c2;

	return Hj;
}

VectorXd Tools::Polar2Cartesian(double rho, double phi) {
  VectorXd result;
  result = VectorXd(2);
  result << rho * cos(phi), rho * sin(phi);
  return result;
}
