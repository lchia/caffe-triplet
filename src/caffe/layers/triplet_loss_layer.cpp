#include <vector>

#include "caffe/layers/triplet_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void TripletLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  CHECK_EQ(bottom[1]->count(1), bottom[2]->count(1))
      << "Inputs must have the same dimension.";

  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels())
      << "Inputs must have the same dimension.";
  CHECK_EQ(bottom[1]->channels(), bottom[2]->channels())
      << "Inputs must have the same dimension.";
  diff_ap_.ReshapeLike(*bottom[0]);
  diff_an_.ReshapeLike(*bottom[0]);
  diff_pn_.ReshapeLike(*bottom[0]);

  diff_sq_ap_.ReshapeLike(*bottom[0]);
  diff_sq_an_.ReshapeLike(*bottom[0]);
  //vector of ones used to sum along channels
  summer_vec_.Reshape(bottom[0]->channels(), 1,1,1);
  for (int i = 0; i < bottom[0]->channels(); ++i)
      summer_vec_.mutable_cpu_data()[i] = Dtype(1);

  dist_binary_.Reshape(bottom[0]->count(1), 1,1,1);
  for (int i = 0; i < bottom[0]->count(1); ++i)
      summer_vec_.mutable_cpu_data()[i] = Dtype(1);
}

template <typename Dtype>
void TripletLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count(1);
  const Dtype* sampleW = bottom[3]->cpu_data();
  caffe_sub(
      count,
      bottom[0]->cpu_data(),  // anchor
      bottom[1]->cpu_data(),  // positive
      diff_ap_.mutable_cpu_data()); // anchor - positive
  caffe_sub(
      count,
      bottom[0]->cpu_data(),  // anchor
      bottom[2]->cpu_data(),  // negative
      diff_an_.mutable_cpu_data()); // anchor - negative
  caffe_sub(
      count,
      bottom[1]->cpu_data(),  // positve
      bottom[2]->cpu_data(),  // negative
      diff_pn_.mutable_cpu_data()); // positive - negative
  const int channels = bottom[0]->channels();
  Dtype margin = this->layer_param_.triplet_loss_param().margin();

  Dtype loss(0.0);
  for (int i = 0; i < bottom[0]->count(1); ++i) {
      dist_sq_ap_.mutable_cpu_data()[i] = caffe_cpu_dot(channels,
          diff_ap_.cpu_data() + (i*channels), diff_ap_.cpu_data() + (i*channels));
      dist_sq_an_.mutable_cpu_data()[i] = caffe_cpu_dot(channels,
          diff_an_.cpu_data() + (i*channels), diff_an_.cpu_data() + (i*channels));
      Dtype mdist = sampleW[i]*std::max(margin + dist_sq_ap_.cpu_data()[i] - dist_sq_an_.cpu_data()[i], Dtype(0.0));
      loss += mdist;
      if (mdist == Dtype(0)) {
          //dist_binary_.mutable_cpu_data()[i] = Dtype(0);
          //Prepare for backward pass
          caffe_set(channels, Dtype(0), diff_ap_.mutable_cpu_data() + (i*channels));
          caffe_set(channels, Dtype(0), diff_an_.mutable_cpu_data() + (i*channels));
          caffe_set(channels, Dtype(0), diff_pn_.mutable_cpu_data() + (i*channels));
      }
  }
  loss = loss / static_cast<Dtype>(bottom[0]->num()) / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}


template <typename Dtype>
void TripletLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  //Dtype margin = this->layer_param_.triplet_loss_param().margin();
  const Dtype* sampleW = bottom[3]->cpu_data();
  for (int i = 0; i < 3; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i < 2) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / static_cast<Dtype>(bottom[i]->num());
      int num = bottom[i]->num();
      int channels = bottom[i]->channels();
      for (int j = 0; j < num; j++) {
          Dtype* bout = bottom[i]->mutable_cpu_diff();
          if (i == 0) { 
              caffe_cpu_axpby(
                  channels,
                  alpha*sampleW[j],
                  diff_pn_.cpu_data() + (j*channels),
                  Dtype(0.0),
                  bout + (j*channels));
          } else if (i == 1) {
              caffe_cpu_axpby(
                  channels,
                  alpha*sampleW[j],
                  diff_ap_.cpu_data() + (j*channels),
                  Dtype(0.0),
                  bout + (j*channels));
          } else if (i == 2) {
              caffe_cpu_axpby(
                  channels,
                  alpha*sampleW[j],
                  diff_an_.cpu_data() + (j*channels),
                  Dtype(0.0),
                  bout + (j*channels));
          }
      }
    }
  }
}    

#ifdef CPU_ONLY
STUB_GPU(TripletLossLayer);
#endif

INSTANTIATE_CLASS(TripletLossLayer);
REGISTER_LAYER_CLASS(TripletLoss);

}  // namespace caffe
