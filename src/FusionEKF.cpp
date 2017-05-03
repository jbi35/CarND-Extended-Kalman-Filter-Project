#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
        0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
        0, 0.0009, 0,
        0, 0, 0.09;

  H_laser_ << 1.0, 0.0, 0.0, 0.0,
              0.0, 1.0, 0.0, 0.0;

  my_tools_ = Tools::Tools();

}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {

  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    Init(measurement_pack);
    is_initialized_ = true;
    return;
  }
  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  const long current_timestamp = measurement_pack.timestamp_;
  // get dt in seconds
  const float dt = (current_timestamp - previous_timestamp_) / 1.0e6;
  // if measurements are really close, simply skip the second one
  if (dt < 1e-3)
   return;

  previous_timestamp_ = current_timestamp;

  // Update process covariance matrix
  const float dt_2 = dt * dt;
  const float dt_3 = dt_2 * dt;
  const float dt_4 = dt_3 * dt;
  const float var_a = 9;
  ekf_.Q_ <<  dt_4/4*var_a, 0, dt_3/2*var_a, 0,
              0, dt_4/4*var_a, 0, dt_3/2*var_a,
              dt_3/2*var_a, 0, dt_2*var_a, 0,
              0, dt_3/2*var_a, 0, dt_2*var_a;


  // Update state transition matrix
  ekf_.F_(0,2) = dt;
  ekf_.F_(1,3) = dt;

  // kalman filter prediction
  ekf_.Predict();

  /*****************************************************************************
   *  Update
   ****************************************************************************/
  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR)
  {
    // compute radar prediction
    const auto& z = measurement_pack.raw_measurements_;
    Hj_ = my_tools_.CalculateJacobian(ekf_.x_);
    const float px = ekf_.x_(0,0);
    const float py = ekf_.x_(1,0);
    const float vx = ekf_.x_(2,0);
    const float vy = ekf_.x_(3,0);
    const float eps = 1e-5;
    const float rho = sqrt(px * px + py * py);
    const float phi = atan2(py, px);
    const float rho_dot = (px * vx + py * vy) / (eps + rho);
    VectorXd z_pred(3);
    z_pred << rho, phi, rho_dot;
    ekf_.UpdateEKF(z, z_pred, Hj_, R_radar_);
  }
  else if(measurement_pack.sensor_type_ == MeasurementPackage::LASER)
  {
     ekf_.Update(measurement_pack.raw_measurements_, H_laser_, R_laser_);
  }
}

void FusionEKF::Init(const MeasurementPackage &measurement_pack)
{
   VectorXd x;
   x = VectorXd(4);


   MatrixXd P;
   P = MatrixXd(4,4);
   P << 1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1000.0, 0.0,
        0.0, 0.0, 0.0, 1000.0;

   MatrixXd Q;
   Q = MatrixXd(4,4);
   MatrixXd F;

   F = MatrixXd(4,4);
   F << 1.0, 0.0, 1.0, 0.0,
        0.0, 1.0, 0.0, 1.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0;

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    /**
    Convert radar from polar to cartesian coordinates and initialize state.
    */
    const double rho = measurement_pack.raw_measurements_(0,0);
    const double phi = measurement_pack.raw_measurements_(1,0);
    const double rho_dot = measurement_pack.raw_measurements_(2,0);
    VectorXd pos;
    pos = VectorXd(2);
    pos = my_tools_.Polar2Cartesian(rho, phi);

    x << pos(0,0), pos(1,0),0.0,0.0;
  }
  else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
    x <<  measurement_pack.raw_measurements_(0,0),
          measurement_pack.raw_measurements_(1,0),
          0.0, 0.0;
  }
  // call kalman init function
  ekf_.Init(x, P, F, Q);

  // done initializing, no need to predict or update
}
