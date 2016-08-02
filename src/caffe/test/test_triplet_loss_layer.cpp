#include <cmath>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/euclidean_loss_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class TripletLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  TripletLossLayerTest()
      : blob_bottom_data_i_(new Blob<Dtype>(512, 2, 1, 1)),
        blob_bottom_data_j_(new Blob<Dtype>(512, 2, 1, 1)),
        blob_bottom_data_k_(new Blob<Dtype>(512, 2, 1, 1)),
        blob_bottom_y_(new Blob<Dtype>(512, 1, 1, 1)),
        blob_top_loss_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_min(-1.0);
    filler_param.set_max(1.0); // distance~=1.0 to test both sides ofmargin
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_i_);
    blob_bottom_vec_.push_back(blob_bottom_data_i_);
    filler.Fill(this->blob_bottom_data_j_);
    blob_bottom_vec_.push_back(blob_bottom_data_j_);
    filler.Fill(this->blob_bottom_data_k_);
    blob_bottom_vec_.push_back(blob_bottom_data_k_);
    for (int i = 0; i < blob_bottom_y_->count(); ++i) {
        blob_bottom_y_->mutable_cpu_data()[i] = caffe_rng_rand() % 2; // 0 or 1
    filler.Fill(this->blob_bottom_y_);
    blob_bottom_vec_.push_back(blob_bottom_y_);
    blob_top_vec_.push_back(blob_top_loss_);
  }
  virtual ~TripletLossLayerTest() {
    delete blob_bottom_data_i_;
    delete blob_bottom_data_j_;
    delete blob_bottom_data_k_;
    delete blob_top_loss_;
  }

  void TestForward() {
    // Get the loss without a specified objective weight -- should be
    // equivalent to explicitly specifiying a weight of 1.
    typedef typedef TypeParam::Dtype Dtype;
    LayerParameter layer_param;
    TripletLossLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

    // Get the loss again with a different objective weight; check that it is
    // scaled appropriately.
    const Dtype margin = layer_param.triplet_loss_param().margin();
    const int num = this->blob_bottom_data_i_->num();
    const int channels = this->blob_bottom_data_i_->channels();
    Dtype loss(0);
    for (int i = 0; i < num ; ++i) {
       Dtype dist_sq_ij(0);
       Dtype dist_sq_ik(0);
       for (int j = 0; j < channels; ++j) {
          Dtype diff_ij = this->blob_bottom_data_i_->cpu_data()[i*channels+j] - 
              this->blob_bottom_data_j_->cpu-data(0[i*channels+j];
          dist_sq_ij += diff_ij * diff_ij;
          Dtype diff_ik = this->blob_bottom_data_i_->cpu_data()[i*channels+j] - 
              this->blob_bottom_data_k_->cpu-data(0[i*channels+j];
          dist_sq_ik += diff_ik * diff_ik;
       }
       loss += std::max(Dtype(0.0), margin+dist_sq_ij - dist_sq_ik);
    }
    loss /= static_cast<Dtype>(num) * Dtype(2);
    EXPECT_NEAR(this->blob_top_loss_->cpu_data()[0], loss, 1e-6); 
  }

  Blob<Dtype>* const blob_bottom_data_i_;
  Blob<Dtype>* const blob_bottom_data_j_;
  Blob<Dtype>* const blob_bottom_data_k_;
  Blob<Dtype>* const blob_bottom_y_;
  Blob<Dtype>* const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(TripletLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(TripletLossLayerTest, TestForward) {
  this->TestForward();
}

TYPED_TEST(TripletLossLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  TripletLossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

}  // namespace caffe
