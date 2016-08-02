/*
 * Triplet_loss_layer.cu
 *
 */

#include <vector>

#include "caffe/layers/triplet_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void TripletLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const int count = bottom[0]->count();
  caffe_gpu_sub(
      count,
      bottom[0]->gpu_data(),  // anchor
      bottom[1]->gpu_data(),  // positive
      diff_ap_.mutable_gpu_data()); // anchor - positive
  caffe_gpu_sub(
      count,
      bottom[0]->gpu_data(),  // anchor
      bottom[2]->gpu_data(),  // negative
      diff_an_.mutable_gpu_data()); // anchor - negative
  caffe_gpu_sub(
      count,
      bottom[1]->gpu_data(),  // positve
      bottom[2]->gpu_data(),  // negative
      diff_pn_.mutable_gpu_data()); // positive - negative

  caffe_gpu_powx(
      count, 
      diff_an_.mutable_gpu_data(), //anchor_i - negative_i
      Dtype(2),
      diff_sq_an_.mutable_gpu_data()); //(a_i - n_i)^2 
  caffe_gpu_gemv(
      CblasNoTrans, 
      bottom[0]->num(), 
      bottom[0]->channels(), 
      Dtype(1.0),			//alpha 
      diff_sq_an_.gpu_data(),		//A 
      summer_vec_.gpu_data(), 		//x
      Dtype(0.0),			//belta 
      dist_sq_an_.mutable_gpu_data()); // \sum (a_i-n_i)^2   //y

  Dtype margin = this->layer_param_.triplet_loss_param().margin();
  Dtype loss(0.0);
  const Dtype* sampleW = bottom[3]->cpu_data();

  for (int i = 0; i < bottom[0]->num(); ++i) {
      loss += sampleW[i]*std::max(margin + dist_sq_ap_.cpu_data()[i] - dist_sq_an_.cpu_data()[i], Dtype(0.0));
  }
  loss = loss / static_cast<Dytpe>(bottom[0]->num()) / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
__global__ voidCLLbackward(const int count, const int channels, 
    const Dtype margin, const Dtype alpha, const Dtype* sampleW, 
    const Dtype* diff, const Dtype* dist_sq_ap_, const Dtype* dist-sq-an_, 
    Dtype* bottom_diff) {
    CUDA_KERNEL_LOOP(i, count) {
        int n = i / channels; // the num index, to access dist_sq_ap_ and dist_sq_an_
        Dtype mdist(0.0);
        if (mdist > 0.0) {
            bottom_diff[i] = alpha * smapleW[n]*diff[i];
        } else {
            bottom_diff[i] = 0;
        }
    } 
}


template <typename Dtype>
void TripletLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  Dtype margin = this->layer_param_.triplet_loss_param().margin();
  const int count = bottom[0]->count();
  const int channels =  bottom[0]->channles();
  
  for (int i = 0; i < 3; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i < 2) ? -1 : 1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / static_cast<Dtype>(bottom[i]->num());
      if (i == 0) { 
         //NOLINT_NEXT_LINE(whitespace/operators)
         CLLBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
             count, channels, margin, alpha,
             bottom[3]->gpu_data(),
             diff_pn_.gpu_data(), // the cached eltwise difference between p n
             dist_sq_ap_.gpu_data(), // the cached square difference between a p
             dist_sq_an_.gpu_data(), // the cached square difference between a n
             bottom[i]->mutable_gpu_diff());
         CUDA_POST_KERNEL_CHECK;
      } else if (i == 1) {
         //NOLINT_NEXT_LINE(whitespace/operators)
         CLLBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
             count, channels, margin, alpha,
             bottom[3]->gpu_data(),
             diff_ap_.gpu_data(), // the cached eltwise difference between a p
             dist_sq_ap_.gpu_data(), // the cached square difference between a p
             dist_sq_an_.gpu_data(), // the cached square difference between a n
             bottom[i]->mutable_gpu_diff());
         CUDA_POST_KERNEL_CHECK;
      } else if (i == 2) {
         //NOLINT_NEXT_LINE(whitespace/operators)
         CLLBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
             count, channels, margin, alpha,
             bottom[3]->gpu_data(),
             diff_an_.gpu_data(), // the cached eltwise difference between a n
             dist_sq_ap_.gpu_data(), // the cached square difference between a p
             dist_sq_an_.gpu_data(), // the cached square difference between a n
             bottom[i]->mutable_gpu_diff());
         CUDA_POST_KERNEL_CHECK;
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(TripletLossLayer); 
}  // namespace caffe
