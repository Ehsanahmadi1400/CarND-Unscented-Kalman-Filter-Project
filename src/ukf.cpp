#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 2;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 2;

  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  /**
   * End DO NOT MODIFY section for measurement noise values
   */

  /**
   * TODO: Complete the initialization. See ukf.h for other member properties.
   * Hint: one or more values initialized above might be wildly off...
   */
  n_x_ = 5;
  n_aug_ = 7;
  lambda_ = 3 - n_aug_;
  is_initialized_ = false;
  weights_ = VectorXd(2*n_aug_+1);
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

  x_ = VectorXd(5);
  x_.setZero();

  P_ = MatrixXd(5, 5);
  P_ << 1, 0, 0, 0, 0,  // later will change it to Identity matrix directly
        0, 1, 0, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 0, 1, 0,
        0, 0, 0, 0, 1;
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Make sure you switch between lidar and radar
   * measurements.
   */
  // read data either from laser or radar and initiliaze x and P
  if(!is_initialized_) {

    time_us_ = meas_package.timestamp_;
    std::cout << "start  initialization ... " << std::endl;

    if(meas_package.sensor_type_ == MeasurementPackage::RADAR) {

      float rho    = meas_package.raw_measurements_(0);
      float theta  = meas_package.raw_measurements_(1);

      x_(0) = rho*cos(theta);
      x_(1) = rho*sin(theta);

    }
    else if(meas_package.sensor_type_ == MeasurementPackage::LASER) {

      x_(0) = meas_package.raw_measurements_(0);
      x_(1) = meas_package.raw_measurements_(1);

    }

    is_initialized_ = true;

    return;
  }

  double delta_t = (meas_package.timestamp_ - time_us_)/ 1000000.0;

  Prediction(delta_t);

  if(meas_package.sensor_type_ == MeasurementPackage::LASER) {

    UpdateLidar(meas_package);
  }
  else if(meas_package.sensor_type_ == MeasurementPackage::RADAR) {

    UpdateRadar(meas_package);

  }
  time_us_ = meas_package.timestamp_;

}

void UKF::Prediction(double delta_t) {
  /**
   * TODO: Complete this function! Estimate the object's location.
   * Modify the state vector, x_. Predict sigma points, the state,
   * and the state covariance matrix.
   */

  //create augmented mean vector
  VectorXd x_aug = VectorXd(n_aug_);

  //create augmented state covariance
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);

  //create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2*n_aug_ + 1);

  //feeding data to x_aug
  x_aug.head(5) = x_;
  x_aug(5) = 0;
  x_aug(6) = 0;

  // MatrixXd Q = MatrixXd(2, 2);
  // Q << std_a_*std_a_, 0,
  //      0, std_yawdd_*std_yawdd_;

  P_aug.fill(0.0);
  P_aug.topLeftCorner(5, 5) = P_;
  P_aug(5, 5) = std_a_*std_a_;
  P_aug(6, 6) = std_yawdd_*std_yawdd_;

  MatrixXd A = P_aug.llt().matrixL();

  //creating sigma points
  Xsig_aug.col(0) = x_aug;

  for(int i=0; i < n_aug_; i++) {

    Xsig_aug.col(i+1)        = x_aug + sqrt(lambda_ + n_aug_) * A.col(i);
    Xsig_aug.col(i+n_aug_+1) = x_aug - sqrt(lambda_ + n_aug_) * A.col(i);
  }

  //predicting sigma points
  for (int i = 0; i< 2*n_aug_+1; i++) {

    //extract values for better readability
    double p_x      = Xsig_aug(0,i);
    double p_y      = Xsig_aug(1,i);
    double v        = Xsig_aug(2,i);
    double yaw      = Xsig_aug(3,i);
    double yawd     = Xsig_aug(4,i);
    double nu_a     = Xsig_aug(5,i);
    double nu_yawdd = Xsig_aug(6,i);

    //predicted state values
    double px_p, py_p;

    //avoid division by zero
    if (fabs(yawd) > 0.001) {
      px_p = p_x + v/yawd * ( sin (yaw + yawd*delta_t) - sin(yaw));
      py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*delta_t) );
    }
    else {
      px_p = p_x + v*delta_t*cos(yaw);
      py_p = p_y + v*delta_t*sin(yaw);
    }
    double v_p = v;
    double yaw_p = yaw + yawd*delta_t;
    double yawd_p = yawd;

    //add noise
    px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
    py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
    v_p = v_p + nu_a*delta_t;

    yaw_p  = yaw_p  + 0.5*nu_yawdd*delta_t*delta_t;
    yawd_p = yawd_p + nu_yawdd*delta_t;

    Xsig_pred_(0,i) = px_p;
    Xsig_pred_(1,i) = py_p;
    Xsig_pred_(2,i) = v_p;
    Xsig_pred_(3,i) = yaw_p;
    Xsig_pred_(4,i) = yawd_p;
  }
  // set weights again! I can later delete it!
  double weight_0 = lambda_/(lambda_+n_aug_);
  weights_(0) = weight_0;
  for (int i=1; i<2*n_aug_+1; i++) {  //2n+1 weights
    double weight = 0.5/(n_aug_+lambda_);
    weights_(i) = weight;
  }

  //predicted state mean
  x_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
    x_ = x_+ weights_(i) * Xsig_pred_.col(i);
  }

  //predicted state covariance matrix
  P_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {

  VectorXd x_diff = Xsig_pred_.col(i) - x_;

  while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
  while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

  P_ = P_ + weights_(i) * x_diff * x_diff.transpose() ;

  }

}

void UKF::UpdateLidar(MeasurementPackage meas_package) {

  /**
   * TODO: Complete this function! Use lidar data to update the belief
   * about the object's position. Modify the state vector, x_, and
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */

  double weight_0 = lambda_/(lambda_+n_aug_);
  weights_(0) = weight_0;
  for (int i=1; i<2*n_aug_+1; i++) {
    double weight = 0.5/(n_aug_+lambda_);
    weights_(i) = weight;
  }

  int n_z_l_ = 2;

  MatrixXd Zsig = MatrixXd(n_z_l_, 2 * n_aug_ + 1);
  VectorXd z_pred = VectorXd(n_z_l_);
  MatrixXd S_l = MatrixXd(n_z_l_,n_z_l_);

  z_pred.fill(0.0);
  S_l.fill(0.0);
  MatrixXd R_l  = MatrixXd(n_z_l_, n_z_l_);
  R_l << std_laspx_*std_laspx_,   0,
         0,   std_laspy_*std_laspy_;

  for(unsigned int j=0; j < 2 * n_aug_ + 1; j++) {

    Zsig(0, j) = Xsig_pred_(0, j);
    Zsig(1, j) = Xsig_pred_(1, j);

  }

  for (int i=0; i < 2*n_aug_+1; i++) {
    z_pred += weights_(i) * Zsig.col(i);
  }

  for(int j=0; j < 2*n_aug_ + 1; j++) {

    VectorXd z_diff = Zsig.col(j) - z_pred;
    S_l += weights_(j)*z_diff * z_diff.transpose();

  }

  S_l += R_l;

  //create matrix for cross correlation Tc
  MatrixXd Tc_l = MatrixXd(n_x_, n_z_l_);
  Tc_l.fill(0.0);

  //storing radar data in each step
  VectorXd z = VectorXd(n_z_l_);
  z = meas_package.raw_measurements_;
//   z << meas_package.raw_measurements_(0),
//        meas_package.raw_measurements_(1);

  for(unsigned int i=0; i < 2*n_aug_ + 1; i++) {

    VectorXd x_diff = Xsig_pred_.col(i) - x_;

    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    VectorXd z_diff = Zsig.col(i) - z_pred;

    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    Tc_l += weights_(i)*x_diff * z_diff.transpose();
  }


  // MatrixXd K_l = MatrixXd(5, 3); no need for this line
  MatrixXd K_l = Tc_l*S_l.inverse();

  //residual
  VectorXd z_diff = z - z_pred;

  while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
  while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

  x_ = x_ + K_l*z_diff;
  P_ = P_ - K_l*S_l*K_l.transpose();

}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use radar data to update the belief
   * about the object's position. Modify the state vector, x_, and
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */

  double weight_0 = lambda_/(lambda_+n_aug_);
  weights_(0) = weight_0;
  for (int i=1; i<2*n_aug_+1; i++) {
    double weight = 0.5/(n_aug_+lambda_);
    weights_(i) = weight;
  }

  int n_z_r_ = 3;
  //create matrix for sigma points in radar measurement space
  MatrixXd Zsig = MatrixXd(n_z_r_, 2 * n_aug_ + 1);

  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z_r_);

  //measurement covariance matrix S
  MatrixXd S_r = MatrixXd(n_z_r_,n_z_r_);

  z_pred.fill(0.0);
  S_r.fill(0.0);
  MatrixXd R  = MatrixXd(3, 3);
  R << std_radr_*std_radr_,   0,   0,
       0, std_radphi_*std_radphi_, 0,
       0,  0,  std_radrd_*std_radrd_;

  for(unsigned int j=0; j < 2 * n_aug_ + 1; j++) {

    // variables:
    float p_x = Xsig_pred_(0, j);
    float p_y = Xsig_pred_(1, j);
    float   v = Xsig_pred_(2, j);
    float yaw = Xsig_pred_(3, j);

    double v1 = cos(yaw)*v;
    double v2 = sin(yaw)*v;

    Zsig(0, j) = sqrt(p_x*p_x + p_y*p_y);
    Zsig(1, j) = atan2(p_y, p_x);
    Zsig(2, j) = (p_x*v1 + p_y*v2)/ sqrt(p_x*p_x + p_y*p_y);
  }

  for (int i=0; i < 2*n_aug_+1; i++) {
      z_pred += weights_(i) * Zsig.col(i);
  }

  for(int j=0; j < 2*n_aug_ + 1; j++) {

    VectorXd z_diff = Zsig.col(j) - z_pred;
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    S_r += weights_(j)*z_diff * z_diff.transpose();

  }

  S_r += R;


  //start updating
  MatrixXd Tc_r = MatrixXd(n_x_, n_z_r_);
  Tc_r.fill(0.0);
  //storing radar data in each step
  VectorXd z = VectorXd(n_z_r_);
  z = meas_package.raw_measurements_;
  // z << meas_package.raw_measurements_(0), 
  //      meas_package.raw_measurements_(1),
  //      meas_package.raw_measurements_(2);

  // 2*n_aug + 1 is related to Xsig and Zsig, not the Tc size!
  for(unsigned int i=0; i < 2*n_aug_ + 1; i++) {

    VectorXd z_diff = Zsig.col(i) - z_pred;
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    Tc_r += weights_(i)*x_diff * z_diff.transpose();
  }

  MatrixXd K_r = Tc_r*S_r.inverse();

  VectorXd z_diff = z - z_pred;
  while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
  while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

  x_ = x_ + K_r*z_diff;
  P_ = P_ - K_r*S_r*K_r.transpose();

}
