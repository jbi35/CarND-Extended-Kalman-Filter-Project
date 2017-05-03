#include "kalman_filter.h"
#include <iostream>
using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  Q_ = Q_in;
  I_ = MatrixXd::Identity(P_.rows(), P_.cols());
}

void KalmanFilter::Predict() {
  x_ = F_ * x_;
  P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter::Update(const VectorXd &z, const MatrixXd &H, const MatrixXd &R)
{
  VectorXd z_pred = H*x_;
  UpdateEKF(z, z_pred, H, R);
}

void KalmanFilter::UpdateEKF(const VectorXd &z, const VectorXd &z_pred,
                             const MatrixXd &H, const MatrixXd &R) {
  // if z_pred and H are given, the update equations are the same for the KF and the EKF
  // and hence only implemented once
  const VectorXd y = z - z_pred;
  const MatrixXd Ht = H.transpose();
  const MatrixXd S = H * P_ * Ht + R;
  const MatrixXd K = P_ * Ht * S.inverse();

  //new estimate
  x_ = x_ + (K * y);
  P_ = (I_ - K * H) * P_;
}
